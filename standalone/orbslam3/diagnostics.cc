#include "diagnostics.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace neon_diag {
namespace {
std::mutex mutex;
std::ofstream init, events;
std::string directory;
std::map<double,long> sources;
std::map<long,unsigned long> epochs;
unsigned long sequence=0, reset_count=0, request_count=0, pending_request=0;
double input_time=-1;
std::chrono::steady_clock::time_point started;
thread_local InitRecord* current=nullptr;
thread_local Geometry geometry;
thread_local std::string reset_reason;
thread_local bool recorded=false;
const double missing=std::numeric_limits<double>::quiet_NaN();
std::string number(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream s; s<<std::setprecision(17)<<v; return s.str();
}
std::string quoted(const std::string& value) {
    std::string result="\"";
    for (char c:value) {
        if(c=='\\' || c=='\"') result+='\\';
        if(c=='\n') result+="\\n"; else result+=c;
    }
    return result+'\"';
}
long source(double t) { auto it=sources.find(t); return it==sources.end() ? -1 : it->second; }
void event(const std::string& kind,long map,double time,const std::string& reason,
           int imu,int stage,double scale,const std::string& gyro,const std::string& accel,int state=-1,int queue=-1) {
    if(!events.is_open()) return;
    const long id=source(time);
    events<<"{\"sequence\":"<<++sequence<<",\"source_frame_id\":"<<(id<0?"null":std::to_string(id))
          <<",\"elapsed_wall_s\":"<<number(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count())
          <<",\"slam_time_s\":"<<(time<0?"null":number(time))
          <<",\"input_source_frame_id\":"<<(source(input_time)<0?"null":std::to_string(source(input_time)))
          <<",\"map_id\":"<<(map<0?"null":std::to_string(map))
          <<",\"map_epoch\":"<<(map<0?"null":std::to_string(epochs[map]))
          <<",\"actual_reset_index\":"<<reset_count<<",\"request_id\":"<<(pending_request?std::to_string(pending_request):"null")
          <<",\"event\":"<<quoted(kind)<<",\"reason\":"<<quoted(reason)
          <<",\"imu_initialized\":"<<(imu<0?"null":std::to_string(imu))
          <<",\"inertial_ba_stage\":"<<(stage<0?"null":std::to_string(stage))
          <<",\"scale_estimate\":"<<number(scale)
          <<",\"gyro_bias\":"<<gyro<<",\"accel_bias\":"<<accel
          <<",\"tracking_state\":"<<(state<0?"null":std::to_string(state))
          <<",\"queued_keyframes\":"<<(queue<0?"null":std::to_string(queue))<<"}\n";
    events.flush();
}
}
void Open(const std::string& dir,const std::vector<double>& times,const std::vector<long>& ids) {
    std::lock_guard<std::mutex> lock(mutex);
    if(times.size()!=ids.size()) throw std::runtime_error("Diagnostic timestamp/ID mismatch");
    directory=dir;
    sources.clear(); epochs.clear(); sequence=reset_count=request_count=pending_request=0; input_time=-1;
    started=std::chrono::steady_clock::now();
    for(size_t i=0;i<times.size();++i) sources.emplace(times[i],ids[i]);
    init.open(dir+"/init_diagnostics.csv"); events.open(dir+"/map_events.jsonl");
    if(!init || !events) throw std::runtime_error("Cannot open diagnostic logs");
    init<<"source_frame_id,slam_time_s,initial_reference_frame_id,reference_age_s,timeout_age_s,reference_keypoints,current_keypoints,initialization_matches,two_view_model,geometric_inliers,good_points,triangulated_points,parallax_deg,similar_hypotheses,geometry_result,initialization_result\n"<<std::setprecision(17);
}
void Close() { std::lock_guard<std::mutex> lock(mutex); init.close(); events.close(); }
std::string SnapshotDirectory() { std::lock_guard<std::mutex> lock(mutex); return directory; }
unsigned long MapEpoch(long map) { std::lock_guard<std::mutex> lock(mutex); return epochs[map]; }
long Source(double t) { std::lock_guard<std::mutex> lock(mutex); return source(t); }
void Input(double t) { std::lock_guard<std::mutex> lock(mutex); input_time=t; recorded=false; }
void EndFrame(double time,int state,int keys) {
    if(recorded) return;
    InitScope row(time,keys,false,-1,-1,-1);
    row.record.result=state==2 ? "TRACKING" : (state==3 ? "RECENTLY_LOST" : (state==4 ? "LOST" : "NO_INITIALIZATION_CALL"));
}
void ResetReason(const std::string& reason) { reset_reason=reason; }
void Event(const std::string& kind,long map,double time,const std::string& reason,int imu,int stage,double scale,const std::string& gyro,const std::string& accel) {
    std::lock_guard<std::mutex> lock(mutex); event(kind,map,time,reason,imu,stage,scale,gyro,accel);
}
void RequestReset() {
    std::lock_guard<std::mutex> lock(mutex);
    pending_request=++request_count;
    event("RESET_REQUEST",-1,input_time,reset_reason,-1,-1,missing,"null","null"); reset_reason.clear();
}
void BeginReset(long map,double time) {
    std::lock_guard<std::mutex> lock(mutex); ++epochs[map]; ++reset_count;
    event("RESET_APPLIED",map,time,"Tracking::ResetActiveMap",-1,-1,missing,"null","null");
    pending_request=0;
}
void TrackingStatus(double time,long map,int state,int queue,int imu,int stage) {
    std::lock_guard<std::mutex> lock(mutex);
    event("TRACK_STATUS",map,time,"",imu,stage,missing,"null","null",state,queue);
}
Geometry& TwoView() { return geometry; }
void InitResult(const std::string& value) { if(current) current->result=value; }
InitScope::InitScope(double time,int keys,bool ready,double reference,int reference_keys,double last_time) {
    record.time=time; record.current_keys=keys;
    record.result=ready?"TWO_VIEW_FAILED":"REFERENCE_KEYPOINTS_LOW";
    if(ready) {record.reference=reference;record.reference_keys=reference_keys;record.age=time-reference;record.timeout_age=last_time-reference;}
    geometry=Geometry{}; current=&record;
}
InitScope::~InitScope() {
    record.geometry=geometry; current=nullptr; recorded=true;
    std::lock_guard<std::mutex> lock(mutex); if(!init.is_open()) return;
    auto integer=[](long v){return v<0?std::string():std::to_string(v);};
    auto value=[](double v){return v<0?std::string():number(v);};
    const auto& g=record.geometry;
    init<<source(record.time)<<','<<record.time<<','<<integer(source(record.reference))<<','<<value(record.age)<<','<<value(record.timeout_age)
        <<','<<integer(record.reference_keys)<<','<<record.current_keys<<','<<integer(record.matches)<<','<<g.model<<','<<integer(g.inliers)
        <<','<<integer(g.good)<<','<<integer(g.triangulated)<<','<<value(g.parallax)<<','<<integer(g.similar)<<','<<g.reason<<','<<record.result<<'\n';
    init.flush(); events.flush();
}
std::string Vector(double x,double y,double z) { return '['+number(x)+','+number(y)+','+number(z)+']'; }
}
