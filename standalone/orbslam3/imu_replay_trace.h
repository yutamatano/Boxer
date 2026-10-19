#pragma once
// Replay-only observability. No live optimizer or pinned g2o source is changed.
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/solver.h"
#include <boost/property_tree/ptree.hpp>
#include <vector>

namespace neon_diag {
inline double NumericGradientNorm(g2o::SparseOptimizer& optimizer) {
    double squared = 0.;
    const double eps = 1e-6;
    for (const auto& pair : optimizer.vertices()) {
        auto v = static_cast<g2o::OptimizableGraph::Vertex*>(pair.second);
        if (v->fixed()) continue;
        std::vector<double> delta(v->dimension(), 0.);
        for (int j = 0; j < v->dimension(); ++j) {
            delta[j] = eps;
            v->push(); v->oplus(delta.data());
            optimizer.computeActiveErrors();
            const double plus = optimizer.activeRobustChi2();
            v->pop();
            delta[j] = -eps;
            v->push(); v->oplus(delta.data());
            optimizer.computeActiveErrors();
            const double minus = optimizer.activeRobustChi2();
            v->pop(); delta[j] = 0.;
            const double gradient = (plus - minus) / (2 * eps);
            squared += gradient * gradient;
        }
    }
    optimizer.computeActiveErrors();
    return std::sqrt(squared);
}

class ReplayLevenberg : public g2o::OptimizationAlgorithmLevenberg {
public:
    boost::property_tree::ptree trace;
    explicit ReplayLevenberg(g2o::Solver* solver)
        : g2o::OptimizationAlgorithmLevenberg(solver) {}
    SolverResult solve(int iteration, bool online = false) override {
        _optimizer->computeActiveErrors();
        const double before = _optimizer->activeRobustChi2();
        const SolverResult status = g2o::OptimizationAlgorithmLevenberg::solve(iteration, online);
        // Save the step that the native solver last attempted. If rejected this
        // is deliberately labelled attempted, not the accepted state update.
        double attempted = 0., gradient_rhs = 0.;
        for (size_t j = 0; j < _solver->vectorSize(); ++j) {
            attempted += _solver->x()[j] * _solver->x()[j];
            gradient_rhs += _solver->b()[j] * _solver->b()[j];
        }
        _optimizer->computeActiveErrors();
        const double after = _optimizer->activeRobustChi2();
        const bool accepted = after < before;
        boost::property_tree::ptree row;
        row.put("iteration", iteration); row.put("chi2_before", before);
        row.put("chi2_after", after); row.put("accepted", accepted);
        row.put("inner_trials", levenbergIteration());
        row.put("rejected_trials", levenbergIteration() - int(accepted));
        row.put("attempted_step_l2", std::sqrt(attempted));
        row.put("accepted_step_l2", accepted ? std::sqrt(attempted) : 0.);
        for (const auto& pair : _optimizer->vertices())
            if (auto scale=dynamic_cast<ORB_SLAM3::VertexScale*>(pair.second))
                row.put("scale_after",scale->estimate());
        row.put("gradient_rhs_l2_before", std::sqrt(gradient_rhs));
        row.put("numeric_gradient_l2_after", NumericGradientNorm(*_optimizer));
        row.put("lambda_after", currentLambda()); row.put("status", int(status));
        row.put("small_improvement_streak", _nBad);
        trace.push_back({"", row});
        return status;
    }
};
}
