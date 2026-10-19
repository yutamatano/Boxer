#include "diagnostics.h"
#include <stdexcept>
int main(int argc,char** argv) {
    if(argc!=2) throw std::runtime_error("Provide output directory");
    neon_diag::Open(argv[1],{10.,10.1,10.2},{9988,9989,9990});
    neon_diag::Event("MAP_CREATED",0);
    neon_diag::Input(10.);
    { neon_diag::InitScope row(10.,150,false,-1,-1,-1); row.record.result="REFERENCE_SET"; }
    neon_diag::ResetReason("test_reset"); neon_diag::RequestReset(); neon_diag::BeginReset(0,10.);
    neon_diag::Input(10.1);
    { neon_diag::InitScope row(10.1,200,true,10.,150,10.);
      row.record.matches=101; row.record.result="TWO_VIEW_FAILED";
      auto& geometry=neon_diag::TwoView(); geometry.model="F"; geometry.inliers=100;
      geometry.triangulated=60; geometry.parallax=0.5; geometry.reason="PARALLAX_LOW"; }
    neon_diag::TrackingStatus(10.1,0,1,3,0,0);
    neon_diag::Event("MAP_CREATED",1);
    neon_diag::Event("IMU_OPTIMIZATION_ESTIMATE",1,10.,"",1,0,1.2,
                     neon_diag::Vector(1.,2.,3.),neon_diag::Vector(4.,5.,6.));
    neon_diag::Event("NEGATIVE_SCALE_TEST",1,10.,"",0,0,-1.);
    neon_diag::ResetReason("second_reset"); neon_diag::RequestReset(); neon_diag::BeginReset(0,10.1);
    neon_diag::EndFrame(10.1,1,200); // Initialization row already exists: do not duplicate it.
    neon_diag::Input(10.2); neon_diag::EndFrame(10.2,2,250);
    neon_diag::Close();
    return 0;
}
