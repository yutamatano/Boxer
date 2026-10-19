#include "imu_snapshot.h"
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=3 && !(argc==4 && std::string(argv[3])=="--correct-scale-jacobian")) {std::cerr<<"Usage: replay_imu_initialization SNAPSHOT.json RESULT.json [--correct-scale-jacobian]\n";return 2;}
    try {return neon_diag::ReplayImu(argv[1],argv[2],argc==4);}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
