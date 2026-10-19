#pragma once
#include "G2oTypes.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include <boost/property_tree/ptree.hpp>
namespace neon_diag {
bool VisualEnabled(ORB_SLAM3::Map*,double time);
void VisualMap(ORB_SLAM3::Map*,const std::string& stage,double time);
void VisualFrame(ORB_SLAM3::Frame*,const std::string& stage);
void VisualInitial(ORB_SLAM3::Map*,double reference,double current,const Sophus::SE3f&,const std::vector<int>&,const std::vector<cv::Point3f>&);
struct VisualSnapshot {
    std::vector<int> vertex_ids;
    std::vector<g2o::OptimizableGraph::Edge*> edges;
    std::string path;
    void Vertex(g2o::OptimizableGraph::Vertex* v) {vertex_ids.push_back(v->id());}
    void Edge(g2o::OptimizableGraph::Edge* e) {edges.push_back(e);}
    void Capture(g2o::SparseOptimizer&,ORB_SLAM3::Map*,double time,const std::string& stage,int iterations,double lambda,bool stop);
    void Result(g2o::SparseOptimizer&,int actual_iterations,bool stop);
};
int ReplayVisual(const std::string& input,const std::string& output,bool anchor_scale);
}
