#include "imu_snapshot.h"
#include "diagnostics.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "KeyFrame.h"
#include <boost/property_tree/json_parser.hpp>
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <cstdlib>
using namespace ORB_SLAM3;
using boost::property_tree::ptree;
namespace neon_diag {
namespace {
// Experimental replay-only condition. The live SLAM estimator still constructs
// the unchanged upstream EdgeInertialGS in Optimizer.cc.
class CorrectScaleEdge : public EdgeInertialGS {
public:
    explicit CorrectScaleEdge(IMU::Preintegrated* p):EdgeInertialGS(p) {}
    void linearizeOplus() override {
        EdgeInertialGS::linearizeOplus();
        _jacobianOplus[7]*=static_cast<const VertexScale*>(_vertices[7])->estimate();
    }
};
std::atomic<unsigned long> serial{0};
thread_local std::string last_snapshot;
template<class D> ptree matrix(const Eigen::MatrixBase<D>& m) {
    ptree a;
    for(int r=0;r<m.rows();++r) for(int c=0;c<m.cols();++c) {
        if(!std::isfinite(m(r,c))) throw std::runtime_error("Nonfinite snapshot matrix");
        ptree x; std::ostringstream s; s<<std::setprecision(17)<<m(r,c); x.put_value(s.str()); a.push_back({"",x});
    } return a;
}
template<class D> void load(const ptree& p, Eigen::MatrixBase<D>& m) {
    if(p.size()!=static_cast<size_t>(m.size())) throw std::runtime_error("Snapshot matrix dimensions");
    auto it=p.begin(); for(int r=0;r<m.rows();++r) for(int c=0;c<m.cols();++c) {
        m(r,c)=it++->second.get_value<double>();
        if(!std::isfinite(m(r,c))) throw std::runtime_error("Nonfinite snapshot input");
    }
}
ptree bias(const IMU::Bias& b) { Eigen::Matrix<float,6,1> v; v<<b.bax,b.bay,b.baz,b.bwx,b.bwy,b.bwz; return matrix(v); }
IMU::Bias readBias(const ptree& p) { Eigen::Matrix<float,6,1> v; load(p,v); return IMU::Bias(v[0],v[1],v[2],v[3],v[4],v[5]); }
ptree vertex(g2o::OptimizableGraph::Vertex* v) {
    ptree p; p.put("id",v->id());p.put("fixed",v->fixed());
    if(auto x=dynamic_cast<VertexPose*>(v)) {p.put("type","pose");p.add_child("Rwb",matrix(x->estimate().Rwb));p.add_child("twb",matrix(x->estimate().twb));
        if(!x->estimate().Rcw.empty()) p.add_child("camera_center",matrix(-x->estimate().Rcw[0].transpose()*x->estimate().tcw[0]));
        ptree cams;for(size_t i=0;i<x->estimate().Rbc.size();++i) {ptree cam;cam.add_child("Rbc",matrix(x->estimate().Rbc[i]));cam.add_child("tbc",matrix(x->estimate().tbc[i]));cams.push_back({"",cam});}p.add_child("camera_extrinsics",cams);
    }
    else if(auto x=dynamic_cast<VertexVelocity*>(v)) {p.put("type","velocity");p.add_child("value",matrix(x->estimate()));}
    else if(auto x=dynamic_cast<VertexGyroBias*>(v)) {p.put("type","gyro");p.add_child("value",matrix(x->estimate()));}
    else if(auto x=dynamic_cast<VertexAccBias*>(v)) {p.put("type","accel");p.add_child("value",matrix(x->estimate()));}
    else if(auto x=dynamic_cast<VertexGDir*>(v)) {p.put("type","gravity");p.add_child("value",matrix(x->estimate().Rwg));}
    else if(auto x=dynamic_cast<VertexScale*>(v)) {p.put("type","scale");p.put("value",x->estimate());}
    else throw std::runtime_error("Unsupported snapshot vertex");return p;
}
ptree result(g2o::SparseOptimizer& o) {
    o.computeActiveErrors();ptree p;p.put("chi2",o.activeChi2());ptree vs;
    for(auto& kv:o.vertices())vs.push_back({"",vertex(static_cast<g2o::OptimizableGraph::Vertex*>(kv.second))});
    p.add_child("vertices",vs);return p;
}
}
void ImuSnapshot::Capture(const std::vector<KeyFrame*>& kfs,long map,long maxkf,int iterations,float pg,float pa,bool mono,bool fixed,bool gauss) {
    const auto enabled=std::getenv("NEON_IMU_SNAPSHOTS");
    if(!enabled || std::string(enabled)!="1")return;
    const auto dir=SnapshotDirectory();if(dir.empty())return;
    path=dir+"/imu_snapshot_"+std::to_string(++serial)+".json";
    last_snapshot=path;
    data.put("schema",1);data.put("map_id",map);data.put("map_epoch",MapEpoch(map));data.put("max_kf_id",maxkf);
    data.put("iterations",iterations);data.put("prior_gyro",pg);data.put("prior_accel",pa);data.put("mono",mono);data.put("fixed_velocity",fixed);data.put("bGauss_unused_upstream",gauss);
    data.put("lambda_init",pg!=0 ? 1000 : -1);
    data.put("numeric_encoding","17-digit decimal strings; matrices row-major");
    ptree ks;for(auto k:kfs) {ptree p;p.put("kf_id",k->mnId);p.put("source_frame_id",Source(k->mTimeStamp));p.put("time_s",k->mTimeStamp);p.put("previous_kf",k->mPrevKF?static_cast<long>(k->mPrevKF->mnId):-1);p.put("next_kf",k->mNextKF?static_cast<long>(k->mNextKF->mnId):-1);p.put("bad",k->isBad());p.add_child("gyro_bias",matrix(k->GetGyroBias()));p.add_child("accel_bias",matrix(k->GetAccBias()));
        bool used=false;for(auto e:edges)if(auto ei=dynamic_cast<EdgeInertialGS*>(e))if(ei->vertex(4)->id()==static_cast<int>(k->mnId))used=true;
        p.put("constraint_status",used?"included":(!k->mPrevKF?"no_previous":(k->mnId>static_cast<unsigned long>(maxkf)?"kf_above_max":(k->isBad()?"bad_kf":(k->mPrevKF->mnId>static_cast<unsigned long>(maxkf)?"previous_above_max":"missing_vertex")))));
        ks.push_back({"",p});} data.add_child("keyframes_map_order",ks);
    ptree vs;for(auto v:vertices)vs.push_back({"",vertex(v)});data.add_child("vertices_insertion_order",vs);
    ptree es;for(auto e:edges) {ptree p;p.put("internal_id",e->internalId());p.put("level",e->level());
        if(e->robustKernel())throw std::runtime_error("Unexpected IMU robust kernel");
        p.put("robust_kernel","none");ptree ids;for(auto v:e->vertices()){ptree id;id.put_value(v->id());ids.push_back({"",id});}p.add_child("vertices",ids);
        if(auto x=dynamic_cast<EdgeInertialGS*>(e)) {
            p.put("type","imu");p.add_child("information",matrix(x->information()));
            const auto& copy=*x->mpInt; // Observe the same object used by computeError; replay checks detect inconsistencies.
            p.put("dt",x->dt);p.put("preintegration_dt",copy.dT);p.add_child("C",matrix(copy.C));p.add_child("bias",bias(copy.b));p.add_child("updated_bias",bias(x->mpInt->GetUpdatedBias()));
#define SAVE_FIELD(f) p.add_child(#f,matrix(copy.f))
            SAVE_FIELD(dR);SAVE_FIELD(dV);SAVE_FIELD(dP);SAVE_FIELD(JRg);SAVE_FIELD(JVg);SAVE_FIELD(JPg);SAVE_FIELD(JVa);SAVE_FIELD(JPa);
#undef SAVE_FIELD
#define SAVE_CACHED(f) p.add_child("cached_" #f,matrix(x->f))
            SAVE_CACHED(JRg);SAVE_CACHED(JVg);SAVE_CACHED(JPg);SAVE_CACHED(JVa);SAVE_CACHED(JPa);
#undef SAVE_CACHED
            p.add_child("noise_measurement",matrix(copy.Nga.diagonal()));p.add_child("noise_walk",matrix(copy.NgaWalk.diagonal()));
            p.add_child("gravity_vector",matrix(x->gI));
        } else if(auto x=dynamic_cast<EdgePriorAcc*>(e)) {p.put("type","prior_accel");p.add_child("information",matrix(x->information()));p.add_child("measurement",matrix(x->bprior));}
        else if(auto x=dynamic_cast<EdgePriorGyro*>(e)) {p.put("type","prior_gyro");p.add_child("information",matrix(x->information()));p.add_child("measurement",matrix(x->bprior));}
        else throw std::runtime_error("Unsupported snapshot edge");es.push_back({"",p});
    }data.add_child("edges_insertion_order",es);
    boost::property_tree::write_json(path,data);
}
void ImuSnapshot::Result(g2o::SparseOptimizer& o) {if(path.empty())return;ptree r=result(o);boost::property_tree::write_json(path+".result.json",r);}
void MapCheckpoint(Map* map,const std::string& stage) {
    if(last_snapshot.empty())return;
    ptree data,ks;data.put("stage",stage);data.put("map_id",map->GetId());data.put("map_epoch",MapEpoch(map->GetId()));
    for(auto k:map->GetAllKeyFrames()) {ptree p;p.put("kf_id",k->mnId);p.put("source_frame_id",Source(k->mTimeStamp));p.add_child("camera_center",matrix(k->GetCameraCenter()));ks.push_back({"",p});}
    data.add_child("keyframes",ks);boost::property_tree::write_json(last_snapshot+"."+stage+".json",data);
}
int ReplayImu(const std::string& input,const std::string& output,bool correct_scale_jacobian) {
    ptree data;boost::property_tree::read_json(input,data);if(data.get<int>("schema")!=1)throw std::runtime_error("Snapshot schema");
    // Preintegrations outlive the optimizer edges that refer to them.
    std::vector<std::unique_ptr<IMU::Preintegrated>> ints;
    g2o::SparseOptimizer o;
    auto solver=new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolverX(new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>()));
    if(data.get<double>("lambda_init")>0)solver->setUserLambdaInit(data.get<double>("lambda_init"));o.setAlgorithm(solver);
    for(auto& item:data.get_child("vertices_insertion_order")) {auto& p=item.second;auto type=p.get<std::string>("type");g2o::OptimizableGraph::Vertex* v=nullptr;
        if(type=="pose") {auto x=new VertexPose;ImuCamPose pose;load(p.get_child("Rwb"),pose.Rwb);load(p.get_child("twb"),pose.twb);x->setEstimate(pose);v=x;}
#define LOAD_VEC(t, cls) else if(type==t) {auto x=new cls;Eigen::Vector3d a;load(p.get_child("value"),a);x->setEstimate(a);v=x;}
        LOAD_VEC("velocity",VertexVelocity) LOAD_VEC("gyro",VertexGyroBias) LOAD_VEC("accel",VertexAccBias)
#undef LOAD_VEC
        else if(type=="gravity") {Eigen::Matrix3d a;load(p.get_child("value"),a);v=new VertexGDir(a);}
        else if(type=="scale")v=new VertexScale(p.get<double>("value"));else throw std::runtime_error("Unknown snapshot vertex");
        v->setId(p.get<int>("id"));v->setFixed(p.get<bool>("fixed"));if(!o.addVertex(v))throw std::runtime_error("Duplicate snapshot vertex");
    }
    for(auto& item:data.get_child("edges_insertion_order")) {auto& p=item.second;auto type=p.get<std::string>("type");g2o::OptimizableGraph::Edge* e=nullptr;
        if(type=="imu") {auto in=std::unique_ptr<IMU::Preintegrated>(new IMU::Preintegrated);in->dT=p.get<float>("dt");load(p.get_child("C"),in->C);in->b=readBias(p.get_child("bias"));in->SetNewBias(readBias(p.get_child("updated_bias")));
#define READ_FIELD(f) load(p.get_child(#f),in->f)
            READ_FIELD(dR);READ_FIELD(dV);READ_FIELD(dP);
#undef READ_FIELD
#define READ_CACHED(f) load(p.get_child("cached_" #f),in->f)
            READ_CACHED(JRg);READ_CACHED(JVg);READ_CACHED(JPg);READ_CACHED(JVa);READ_CACHED(JPa);
#undef READ_CACHED
            EdgeInertialGS* x=correct_scale_jacobian ? static_cast<EdgeInertialGS*>(new CorrectScaleEdge(in.get())) : new EdgeInertialGS(in.get());Eigen::Matrix<double,9,9> info;load(p.get_child("information"),info);x->setInformation(info);load(p.get_child("gravity_vector"),x->gI);e=x;
            in->dT=p.get<float>("preintegration_dt");
#define READ_FIELD(f) load(p.get_child(#f),in->f)
            READ_FIELD(JRg);READ_FIELD(JVg);READ_FIELD(JPg);READ_FIELD(JVa);READ_FIELD(JPa);
#undef READ_FIELD
            ints.push_back(std::move(in));
        } else {Eigen::Vector3f b;load(p.get_child("measurement"),b);Eigen::Matrix3d info;load(p.get_child("information"),info);
            if(type=="prior_accel") {auto x=new EdgePriorAcc(b);x->setInformation(info);e=x;}
            else if(type=="prior_gyro") {auto x=new EdgePriorGyro(b);x->setInformation(info);e=x;}else throw std::runtime_error("Unknown snapshot edge");}
        size_t i=0;for(auto& id:p.get_child("vertices")){auto v=o.vertex(id.second.get_value<int>());if(!v)throw std::runtime_error("Missing snapshot vertex");e->setVertex(i++,v);}if(!o.addEdge(e))throw std::runtime_error("Invalid snapshot edge");
    }
    o.setVerbose(false);o.initializeOptimization();o.optimize(data.get<int>("iterations"));ptree r=result(o);r.put("scale_jacobian",correct_scale_jacobian?"chain_rule_corrected_replay_only":"upstream");boost::property_tree::write_json(output,r);return 0;
}
}
