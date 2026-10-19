#pragma once
#include "G2oTypes.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include <boost/property_tree/ptree.hpp>
namespace neon_diag {
// Numerical graph copies only; never owns or changes SLAM objects.
struct ImuSnapshot {
    boost::property_tree::ptree data;
    std::vector<g2o::OptimizableGraph::Vertex*> vertices;
    std::vector<g2o::OptimizableGraph::Edge*> edges;
    std::string path;
    void Vertex(g2o::OptimizableGraph::Vertex* v) { vertices.push_back(v); }
    void Edge(g2o::OptimizableGraph::Edge* e) { edges.push_back(e); }
    void Capture(const std::vector<ORB_SLAM3::KeyFrame*>& kfs, long map, long maxkf,
                 int iterations, float priorG, float priorA, bool mono, bool fixed, bool gauss);
    void Result(g2o::SparseOptimizer& optimizer);
};
void MapCheckpoint(ORB_SLAM3::Map* map, const std::string& stage);
int ReplayImu(const std::string& input, const std::string& output, bool correct_scale_jacobian=false, bool numeric_scale_jacobian=false,
              const std::string& warm_start="", bool trace=false);
std::string SnapshotDirectory();
unsigned long MapEpoch(long map);
}
