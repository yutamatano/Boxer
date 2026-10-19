#pragma once
// Observation only. No ORB classes, estimator settings, or data layouts change.
#include <string>
#include <vector>
#include <functional>
#include <limits>
#include <cmath>

namespace neon_diag {
struct Geometry {
    std::string model, reason;
    int inliers=-1, good=-1, triangulated=-1, similar=-1;
    double parallax=-1;
};
struct InitRecord {
    double time=0, reference=-1, age=-1, timeout_age=-1;
    int reference_keys=-1, current_keys=0, matches=-1;
    std::string result;
    Geometry geometry;
};
void Open(const std::string& directory, const std::vector<double>& times,
          const std::vector<long>& source_ids);
void Close();
long Source(double time);
void Input(double time);
void EndFrame(double time,int state,int keys);
void ResetReason(const std::string& reason);
void Event(const std::string& event, long map=-1, double time=-1,
           const std::string& reason="", int imu=-1, int stage=-1,
           double scale=std::numeric_limits<double>::quiet_NaN(), const std::string& gyro="null",
           const std::string& accel="null");
void RequestReset();
void BeginReset(long map, double time);
void TrackingStatus(double time,long map,int state,int queue,int imu,int stage);
struct Finally {
    std::function<void()> callback;
    explicit Finally(std::function<void()> value):callback(value) {}
    ~Finally() { callback(); }
};
Geometry& TwoView();
void InitResult(const std::string& result);
struct InitScope {
    InitRecord record;
    InitScope(double time, int keys, bool ready, double reference,
              int reference_keys, double last_time);
    ~InitScope();
};
std::string Vector(double x,double y,double z);
}
