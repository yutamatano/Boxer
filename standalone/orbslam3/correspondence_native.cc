// Independent benchmark of the unchanged upstream extractor and init matcher.
#include "Frame.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Pinhole.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=6) {std::cerr<<"Usage: correspondence_native camera.yaml images.csv pairs.csv features.bin matches.csv\n";return 2;}
    try {
        cv::setNumThreads(2);cv::FileStorage s(argv[1],cv::FileStorage::READ);
        ORB_SLAM3::ORBextractor extractor(5*int(s["ORBextractor.nFeatures"]),float(s["ORBextractor.scaleFactor"]),int(s["ORBextractor.nLevels"]),int(s["ORBextractor.iniThFAST"]),int(s["ORBextractor.minThFAST"]));
        ORB_SLAM3::Pinhole camera(std::vector<float>{float(s["Camera1.fx"]),float(s["Camera1.fy"]),float(s["Camera1.cx"]),float(s["Camera1.cy"])});
        cv::Mat distortion=cv::Mat::zeros(4,1,CV_32F);
        std::map<long,std::unique_ptr<ORB_SLAM3::Frame>> frames;
        std::ifstream images(argv[2]);std::ofstream features(argv[4],std::ios::binary);std::string line;
        if(!images || !features)throw std::runtime_error("Cannot open feature inputs/output");
        while(std::getline(images,line)) {
            auto comma=line.find(',');long id=std::stol(line.substr(0,comma));auto image=cv::imread(line.substr(comma+1),cv::IMREAD_GRAYSCALE);
            if(image.empty())throw std::runtime_error("Cannot decode image");
            auto f=std::unique_ptr<ORB_SLAM3::Frame>(new ORB_SLAM3::Frame(image,id*.1,&extractor,nullptr,&camera,distortion,0,0));
            const int64_t source=id;const uint32_t n=f->mvKeys.size();features.write(reinterpret_cast<const char*>(&source),8);features.write(reinterpret_cast<const char*>(&n),4);
            for(size_t i=0;i<n;++i) {float values[]={f->mvKeys[i].pt.x,f->mvKeys[i].pt.y,f->mvKeys[i].angle};int32_t octave=f->mvKeys[i].octave;features.write(reinterpret_cast<const char*>(values),12);features.write(reinterpret_cast<const char*>(&octave),4);features.write(reinterpret_cast<const char*>(f->mDescriptors.ptr(i)),32);}
            frames[id]=std::move(f);
            if(frames.size()%100==0)std::cout<<"Native frames "<<frames.size()<<std::endl;
        }
        std::ifstream pairs(argv[3]);std::ofstream matches(argv[5]);if(!pairs || !matches)throw std::runtime_error("Cannot open pair inputs/output");
        matches<<"reference,target,index_reference,index_target,native_count\n";
        ORB_SLAM3::ORBmatcher matcher(.9,true);
        while(std::getline(pairs,line)) {auto comma=line.find(',');long a=std::stol(line.substr(0,comma)),b=std::stol(line.substr(comma+1));auto& f=*frames.at(a);auto& g=*frames.at(b);std::vector<cv::Point2f> previous;for(auto& k:f.mvKeysUn)previous.push_back(k.pt);std::vector<int> indices;
            const int n=matcher.SearchForInitialization(f,g,previous,indices,100);
            if(n==0)matches<<a<<','<<b<<",-1,-1,0\n";
            for(size_t i=0;i<indices.size();++i)if(indices[i]>=0)matches<<a<<','<<b<<','<<i<<','<<indices[i]<<','<<n<<'\n';
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
