// Compare upstream's analytic IMU scale derivative with its actual oplus.
// This is a diagnostic executable. It never changes the estimator.
#include "G2oTypes.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"
#include <iostream>
#include <iomanip>
using namespace ORB_SLAM3;
class InspectEdge : public EdgeInertialGS {
public:
    explicit InspectEdge(IMU::Preintegrated* p):EdgeInertialGS(p) {}
    Eigen::Matrix<double,9,1> scaleJacobian() {
        g2o::JacobianWorkspace workspace;workspace.updateSize(this);workspace.allocate();
        g2o::BaseMultiEdge<9,Vector9d>::linearizeOplus(workspace);
        return _jacobianOplus[7];
    }
};
int main() {
    IMU::Calib calibration(Sophus::SE3f(),.001,.01,.0001,.001);
    IMU::Preintegrated preintegrated(IMU::Bias(0,0,0,0,0,0),calibration);
    preintegrated.IntegrateNewMeasurement(Eigen::Vector3f(0,0,9.81),Eigen::Vector3f::Zero(),.1);
    VertexPose p1,p2;ImuCamPose pose;pose.its=0;pose.bf=0;pose.Rwb0.setIdentity();pose.DR.setIdentity();pose.Rwb.setIdentity();pose.twb.setZero();p1.setEstimate(pose);pose.twb<<.1,.02,.01;p2.setEstimate(pose);
    VertexVelocity v1,v2;v1.setEstimate(Eigen::Vector3d(.2,.1,0));v2.setEstimate(Eigen::Vector3d(.3,.2,.1));
    VertexGyroBias bg;VertexAccBias ba;bg.setEstimate(Eigen::Vector3d::Zero());ba.setEstimate(Eigen::Vector3d::Zero());
    VertexGDir gravity(Eigen::Matrix3d::Identity());VertexScale scale;
    InspectEdge edge(&preintegrated);
    g2o::OptimizableGraph::Vertex* vertices[]={&p1,&v1,&bg,&ba,&p2,&v2,&gravity,&scale};
    for(int i=0;i<8;++i)edge.setVertex(i,vertices[i]);
    std::cout<<"scale,analytic_norm,numeric_norm,relative_error,chain_rule_corrected_relative_error\n"<<std::setprecision(17);
    bool mismatch_observed=false;
    for(double s:{1.,2.,10.}) {
        scale.setEstimate(s);edge.computeError();auto analytic=edge.scaleJacobian();
        const double epsilon=1e-6;double step=epsilon;
        scale.setEstimate(s);scale.oplus(&step);edge.computeError();Eigen::Matrix<double,9,1> plus=edge.error();
        step=-epsilon;scale.setEstimate(s);scale.oplus(&step);edge.computeError();Eigen::Matrix<double,9,1> minus=edge.error();
        auto numeric=((plus-minus)/(2*epsilon)).eval();const double denominator=numeric.norm();
        double error=(analytic-numeric).norm()/denominator;
        double corrected=(s*analytic-numeric).norm()/denominator;
        std::cout<<s<<','<<analytic.norm()<<','<<numeric.norm()<<','<<error<<','<<corrected<<'\n';
        if(s==1. && error>1e-7)return 1;
        if(s!=1. && error>.1)mismatch_observed=true;
        if(corrected>1e-7)return 1;
    }
    return mismatch_observed?0:2;
}
