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
        std::cerr << "Usage: mono_inertial_neon VOCABULARY PREPARED OUTPUT [PACE]\n"
                  << "PACE=1: real-time scheduling; 0: fastest (may starve mapping)\n";
        return 1;
    }
    try {
        const fs::path vocabulary = fs::absolute(argv[1]);
        const fs::path prepared = fs::absolute(argv[2]), output = fs::absolute(argv[3]);
        const double pace = argc == 5 ? std::stod(argv[4]) : 1.0;
        if (!std::isfinite(pace) || pace < 0) throw std::runtime_error("Invalid pace");
        if (fs::exists(output) && !fs::is_empty(output))
            throw std::runtime_error("Output must be an empty directory");
        std::vector<Frame> frames;
        for (const auto& row : rows(prepared / "frames.csv")) {
            if (row.size() != 7) throw std::runtime_error("Invalid frames.csv row");
            frames.push_back({std::stol(row[0]), std::stod(row[3]), row[4]});
        }
        std::vector<Sample> imu;
        for (const auto& row : rows(prepared / "imu.csv")) {
            if (row.size() != 7) throw std::runtime_error("Invalid imu.csv row");
            Sample sample;
            sample.time = std::stod(row[0]);
            float values[6];
            for (int axis = 0; axis < 6; ++axis) {
                values[axis] = std::stof(row[axis + 1]);
                if (!std::isfinite(values[axis]))
                    throw std::runtime_error("Nonfinite IMU measurement");
            }
            sample.gyro = cv::Point3f(values[0], values[1], values[2]);
            sample.accel = cv::Point3f(values[3], values[4], values[5]);
            imu.push_back(sample);
        }
        if (frames.size() < 2 || imu.size() < 2) throw std::runtime_error("Not enough input");
        for (size_t i = 0; i < frames.size(); ++i)
            if (!std::isfinite(frames[i].time) || (i && (frames[i].time <= frames[i-1].time || frames[i].id != frames[i-1].id + 1)))
                throw std::runtime_error("Nonfinite/noncontiguous source frames");
        for (size_t i = 0; i < imu.size(); ++i)
            if (!std::isfinite(imu[i].time) || (i && imu[i].time <= imu[i-1].time))
                throw std::runtime_error("Nonfinite/nonmonotonic IMU times");
        if (imu.front().time >= frames.front().time || imu.back().time <= frames.back().time)
            throw std::runtime_error("IMU does not bracket the selected images");
        fs::create_directories(output);
        // Upstream may write relative diagnostic files. Keep those alongside
        // the run artifacts rather than in the repository working directory.
        fs::current_path(output);
        std::ofstream states(output / "states.csv");
        if (!states) throw std::runtime_error("Cannot write states.csv");
        states << "frame_id,slam_time_s,state,mapped_features,imu_initialized,map_id,track_ms\n" << std::setprecision(17);
        cv::setNumThreads(2);
        ORB_SLAM3::System slam(vocabulary.string(), (prepared / "camera.yaml").string(),
                               ORB_SLAM3::System::IMU_MONOCULAR, false);
        size_t next = 0, metricOK = 0;
        const auto wallStart = std::chrono::steady_clock::now();
        for (size_t i = 0; i < frames.size(); ++i) {
            const Frame& frame = frames[i];
            if (pace > 0) {
                const auto target = wallStart + std::chrono::duration<double>((frame.time - frames.front().time) * pace);
                std::this_thread::sleep_until(target);
            }
            cv::Mat image = cv::imread((prepared / frame.image).string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) throw std::runtime_error("Cannot read " + frame.image);
            std::vector<ORB_SLAM3::IMU::Point> measurements;
            // Include each sample once, including the first sample after this
            // image: upstream keeps that boundary sample for interpolation.
            while (next < imu.size()) {
                const auto& sample = imu[next++];
                measurements.emplace_back(sample.accel, sample.gyro, sample.time);
                if (sample.time >= frame.time) break;
            }
            if (measurements.empty()) throw std::runtime_error("No IMU samples for frame");
            const auto begin = std::chrono::steady_clock::now();
            slam.TrackMonocular(image, frame.time, measurements, frame.image);
            const double milliseconds = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - begin).count();
            const int state = slam.GetTrackingState();
            const auto map = slam.GetNeonMapState();
            size_t mappedFeatures = 0;
            for (auto* point : slam.GetTrackedMapPoints()) if (point && !point->isBad()) ++mappedFeatures;
            states << frame.id << ',' << frame.time << ',' << state << ',' << mappedFeatures << ',' << map.second << ',' << map.first << ',' << milliseconds << '\n';
            states.flush();
            metricOK += state == 2 && map.second;
            if (i % 100 == 0) std::cout << "Neon frame " << i << '/' << frames.size() << " state=" << state << " mapped_features=" << mappedFeatures << " metric=" << map.second << std::endl;
        }
        slam.Shutdown();
        const size_t finalMetric = slam.SaveNeonCameraTrajectory((output / "camera_trajectory.csv").string());
        std::cout << "Final metric camera poses: " << finalMetric << "; runtime metric OK frames: " << metricOK << '/' << frames.size() << std::endl;
        return finalMetric && metricOK ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "Neon VIO: " << error.what() << std::endl;
        return 1;
    }
}
