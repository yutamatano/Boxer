// Offline Neon adapter. ORB-SLAM3 and the linked executable use GPLv3;
// see the upstream LICENSE at https://github.com/UZ-SLAMLab/ORB_SLAM3.
#include <System.h>
#include <MapPoint.h>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;
struct Frame { long id; double time; std::string image; };
struct Sample { double time; cv::Point3f gyro, accel; };

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) result.push_back(value);
    return result;
}

std::vector<std::vector<std::string>> rows(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    std::string line;
    std::getline(file, line);  // preparation writes a fixed schema header
    std::vector<std::vector<std::string>> result;
    while (std::getline(file, line)) if (!line.empty()) result.push_back(fields(line));
    return result;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: neon_standalone VOCABULARY PREPARED OUTPUT [PACE]\n";
        return 1;
    }
    try {
        const fs::path vocabulary = fs::absolute(argv[1]);
        const fs::path prepared = fs::absolute(argv[2]);
        const fs::path output = fs::absolute(argv[3]);
        const double pace = argc == 5 ? std::stod(argv[4]) : 1.;
        if (!std::isfinite(pace) || pace < 0) throw std::runtime_error("Invalid pace");
        if (fs::exists(output) && !fs::is_empty(output))
            throw std::runtime_error("Output must be empty");
        std::vector<Frame> frames;
        for (const auto& row : rows(prepared / "frames.csv")) {
            if (row.size() < 5) throw std::runtime_error("Invalid frames.csv");
            frames.push_back({std::stol(row[0]), std::stod(row[3]), row[4]});
        }
        std::vector<Sample> imu;
        for (const auto& row : rows(prepared / "imu.csv")) {
            if (row.size() != 7) throw std::runtime_error("Invalid imu.csv");
            Sample sample;
            sample.time = std::stod(row[0]);
            float v[6];
            for (int axis=0; axis<6; ++axis) {
                v[axis] = std::stof(row[axis+1]);
                if (!std::isfinite(v[axis])) throw std::runtime_error("Nonfinite IMU");
            }
            sample.gyro = cv::Point3f(v[0],v[1],v[2]);
            sample.accel = cv::Point3f(v[3],v[4],v[5]);
            imu.push_back(sample);
        }
        if (frames.size()<2 || imu.size()<2) throw std::runtime_error("Not enough input");
        for (size_t i=0; i<frames.size(); ++i)
            if (!std::isfinite(frames[i].time) || (i && (frames[i].time<=frames[i-1].time || frames[i].id!=frames[i-1].id+1)))
                throw std::runtime_error("Noncontiguous frames");
        for (size_t i=0; i<imu.size(); ++i)
            if (!std::isfinite(imu[i].time) || (i && imu[i].time<=imu[i-1].time))
                throw std::runtime_error("Nonmonotonic IMU");
        if (imu.front().time>=frames.front().time || imu.back().time<=frames.back().time)
            throw std::runtime_error("IMU must bracket all images");
        fs::create_directories(output);
        fs::current_path(output);
        std::ofstream states(output / "states.csv");
        if (!states) throw std::runtime_error("Cannot open states.csv");
        states << "frame_id,slam_time_s,state,mapped_features,imu_initialized,map_id,track_ms\n" << std::setprecision(17);
        cv::setNumThreads(2);
        ORB_SLAM3::System slam(vocabulary.string(), (prepared / "camera.yaml").string(),
                               ORB_SLAM3::System::IMU_MONOCULAR, false);
        size_t next=static_cast<size_t>(std::lower_bound(imu.begin(),imu.end(),frames.front().time,
            [](const Sample& s,double t) { return s.time<t; })-imu.begin());
        if (next) --next;
        const auto start=std::chrono::steady_clock::now();
        size_t metricOK=0;
        for (size_t i=0; i<frames.size(); ++i) {
            const auto& frame=frames[i];
            if (pace>0)
                std::this_thread::sleep_until(start+std::chrono::duration<double>((frame.time-frames.front().time)*pace));
            cv::Mat image=cv::imread((prepared / frame.image).string(),cv::IMREAD_GRAYSCALE);
            if (image.empty()) throw std::runtime_error("Cannot read image");
            std::vector<ORB_SLAM3::IMU::Point> samples;
            while (next<imu.size()) {
                const auto& s=imu[next++];
                samples.emplace_back(s.accel,s.gyro,s.time);
                if (s.time>=frame.time) break;
            }
            if (samples.empty()) throw std::runtime_error("Missing IMU interval");
            const auto begin=std::chrono::steady_clock::now();
            slam.TrackMonocular(image,frame.time,samples,frame.image);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            const int state=slam.GetTrackingState();
            const auto map=slam.GetNeonMapState();
            size_t features=0;
            for (auto* point : slam.GetTrackedMapPoints()) if (point && !point->isBad()) ++features;
            states << frame.id << ',' << frame.time << ',' << state << ',' << features << ',' << map.second << ',' << map.first << ',' << ms << '\n';
            states.flush();
            metricOK += state==2 && map.second;
            if (i%100==0 || i+1==frames.size())
                std::cout << "Standalone frame " << i+1 << '/' << frames.size() << " state=" << state << " metric=" << map.second << std::endl;
            // Never stop/reset/restart here: upstream owns every map transition.
        }
        slam.Shutdown();
        const auto finalPoses=slam.SaveNeonCameraTrajectory((output / "camera_trajectory.csv").string());
        const auto points=slam.SaveNeonMapPoints((output / "map_points.csv").string());
        std::ofstream report(output / "run.json");
        report << "{\"processed_frames\":" << frames.size() << ",\"runtime_metric_ok_frames\":" << metricOK
               << ",\"final_metric_poses\":" << finalPoses << ",\"map_points\":" << points
               << ",\"pace\":" << pace << ",\"elapsed_seconds\":"
               << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Standalone ORB-SLAM3: " << error.what() << std::endl;
        return 1;
    }
}
