#include "visual_snapshot.h"
#include <iostream>
int main(int argc,char** argv){if(argc!=3 && !(argc==4 && std::string(argv[3])=="--anchor-scale")){std::cerr<<"Usage: replay_visual_ba SNAPSHOT RESULT [--anchor-scale]\n";return 2;}try{return neon_diag::ReplayVisual(argv[1],argv[2],argc==4);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
