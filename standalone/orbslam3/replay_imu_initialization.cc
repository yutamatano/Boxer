#include "imu_snapshot.h"
#include <iostream>
int main(int argc,char** argv) {
    if(argc<3) {std::cerr<<"Usage: replay_imu_initialization SNAPSHOT.json RESULT.json [--correct-scale-jacobian | --numeric-scale-jacobian] [--trace] [--warm-start RESULT.json]\n";return 2;}
    bool corrected=false,numeric=false,trace=false;std::string warm;
    for(int i=3;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--correct-scale-jacobian")corrected=true;
        else if(arg=="--numeric-scale-jacobian")numeric=true;
        else if(arg=="--trace")trace=true;
        else if(arg=="--warm-start" && i+1<argc)warm=argv[++i];
        else {std::cerr<<"Unknown/incomplete option: "<<arg<<'\n';return 2;}
    }
    if(corrected && numeric){std::cerr<<"Choose one scale Jacobian\n";return 2;}
    try {return neon_diag::ReplayImu(argv[1],argv[2],corrected,numeric,warm,trace);}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
