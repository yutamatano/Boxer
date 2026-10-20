"""Generate diagnostic copies with explicitly opt-in comparison policies."""
import argparse
from pathlib import Path


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Pinned diagnostic location changed: {old[:90]!r}")
    return text.replace(old, new)



def visual_optimizer(text):
    import re
    scopes=[('void Optimizer::BundleAdjustment(', 'global_visual_ba', 'nIterations', '-1', 'visualTime'),
            ('void Optimizer::LocalBundleAdjustment(KeyFrame *pKF,', 'local_visual_ba', '10', 'pMap->IsInertial()?100.:-1.', 'pKF->mTimeStamp')]
    for signature,stage,iterations,initial_lambda,timestamp in scopes:
        begin=text.index(signature);end=text.index('\nvoid Optimizer::',begin+10)
        part=text[begin:end]
        brace=part.index('\n{')+2
        declaration='\n    neon_diag::VisualSnapshot visualSnapshot;\n'
        if stage=='global_visual_ba':declaration+='    double visualTime=0; for(auto k:vpKFs) visualTime=std::max(visualTime,k->mTimeStamp);\n'
        part=part[:brace]+declaration+part[brace:]
        part=re.sub(r'optimizer.addVertex\((\w+)\);',r'optimizer.addVertex(\1); visualSnapshot.Vertex(\1);',part)
        part=re.sub(r'optimizer.addEdge\((\w+)\);',r'optimizer.addEdge(\1); visualSnapshot.Edge(\1);',part)
        part=replace(part,'optimizer.initializeOptimization();',
            f'visualSnapshot.Capture(optimizer,pMap,{timestamp},"{stage}",{iterations},{initial_lambda},pbStopFlag && *pbStopFlag);\n    optimizer.initializeOptimization();')
        part=replace(part,f'optimizer.optimize({iterations});',f'int visualActualIterations=optimizer.optimize({iterations});\n    visualSnapshot.Result(optimizer,visualActualIterations,pbStopFlag && *pbStopFlag);')
        text=text[:begin]+part+text[end:]
    begin=text.index('int Optimizer::PoseOptimization(Frame *pFrame)');end=text.index('\nvoid Optimizer::',begin+10)
    part=text[begin:end];brace=part.index('\n{')+2
    part=part[:brace]+"""
    neon_diag::VisualFrame(pFrame,"tracking_pose_before");
    neon_diag::Finally visualFrameResult([pFrame]{neon_diag::VisualFrame(pFrame,"tracking_pose_after");});
"""+part[brace:]
    return text[:begin]+part+text[end:]


def instrument(name, text):
    text = '#include "diagnostics.h"\n' + text
    if name in ['Tracking','LocalMapping','Optimizer']:
        text='#include "visual_snapshot.h"\n'+text
    if name == 'Tracking':
        text = replace(text, 'void Tracking::Track()\n{', '''void Tracking::Track()
{
    neon_diag::Finally diagnosticFrame([this] {
        Map* map=mpAtlas->GetCurrentMap();
        neon_diag::TrackingStatus(mCurrentFrame.mTimeStamp,map ? map->GetId() : -1,mState,
            mpLocalMapper->KeyframesInQueue(),map ? map->isImuInitialized() : -1,
            map ? (map->GetIniertialBA2()?2:(map->GetIniertialBA1()?1:0)) : -1);
    });''')
        text = replace(text, 'void Tracking::MonocularInitialization()\n{', '''void Tracking::MonocularInitialization()
{
    neon_diag::InitScope diagnostic(mCurrentFrame.mTimeStamp, mCurrentFrame.mvKeys.size(), mbReadyToInitializate,
        mbReadyToInitializate ? mInitialFrame.mTimeStamp : -1,
        mbReadyToInitializate ? static_cast<int>(mInitialFrame.mvKeys.size()) : -1,
        mbReadyToInitializate ? mLastFrame.mTimeStamp : -1);''')
        # Restrict edits to this function; do not accidentally alter stereo init.
        begin=text.index('void Tracking::MonocularInitialization()')
        end=text.index('void Tracking::CreateInitialMapMonocular()',begin)
        part=text[begin:end]
        part=replace(part,'mbReadyToInitializate = true;', '''diagnostic.record.result="REFERENCE_SET";
            diagnostic.record.reference=mCurrentFrame.mTimeStamp;
            diagnostic.record.reference_keys=mCurrentFrame.mvKeys.size();
            diagnostic.record.age=diagnostic.record.timeout_age=0;
            mbReadyToInitializate = true;''')
        part=replace(part,'''            mbReadyToInitializate = false;

            return;''','''            diagnostic.record.result=mCurrentFrame.mvKeys.size()<=100 ? "CURRENT_KEYPOINTS_LOW" : "PAIR_TIMEOUT";
            mbReadyToInitializate = false;

            return;''')
        part=replace(part,'if(nmatches<100)', 'diagnostic.record.matches=nmatches;\n        if(nmatches<100)')
        part=replace(part,'''if(nmatches<100)
        {''','''if(nmatches<100)
        {
            diagnostic.record.result="MATCHES_LOW";''')
        part=replace(part,'CreateInitialMapMonocular();','diagnostic.record.result="VISUAL_INITIALIZED";\n            CreateInitialMapMonocular();')
        part=replace(part,'mCurrentFrame.SetPose(Tcw);', 'mCurrentFrame.SetPose(Tcw);\n            neon_diag::VisualInitial(mpAtlas->GetCurrentMap(),mInitialFrame.mTimeStamp,mCurrentFrame.mTimeStamp,Tcw,mvIniMatches,mvIniP3D);')
        text=text[:begin]+part+text[end:]
        text=replace(text,'Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);', 'neon_diag::VisualMap(mpAtlas->GetCurrentMap(),"initial_ba_before",mCurrentFrame.mTimeStamp);\n    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);\n    neon_diag::VisualMap(mpAtlas->GetCurrentMap(),"initial_ba_after",mCurrentFrame.mTimeStamp);')
        text=replace(text,'// Scale initial baseline', 'neon_diag::VisualMap(mpAtlas->GetCurrentMap(),"normalization_before",mCurrentFrame.mTimeStamp);\n    // Scale initial baseline')
        text=replace(text,'pKFcur->mPrevKF = pKFini;', 'neon_diag::VisualMap(mpAtlas->GetCurrentMap(),"normalization_after",mCurrentFrame.mTimeStamp);\n        pKFcur->mPrevKF = pKFini;')
        text=replace(text,'Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);',
                     'neon_diag::InitResult("POST_BA_FAILED");\n        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);')
        text=replace(text,'mpAtlas->clearMap();',
                     'mpAtlas->clearMap();\n    neon_diag::BeginReset(pMap->GetId(), mCurrentFrame.mTimeStamp);')
        # An unambiguous call-site reason for every tracking reset request.
        lines=text.splitlines()
        result=[]
        for line_no,line in enumerate(lines,1):
            if 'mpSystem->ResetActiveMap();' in line:
                indent=line[:len(line)-len(line.lstrip())]
                # Keep unbraced control flow intact by using one compound statement.
                line=line.replace('mpSystem->ResetActiveMap();', '{ neon_diag::ResetReason("Tracking_reset_site_'+str(line_no)+'"); mpSystem->ResetActiveMap(); }')
            result.append(line)
        text='\n'.join(result)+'\n'
    elif name == 'TwoViewReconstruction':
        text=replace(text,'if(SH+SF == 0.f) return false;', 'if(SH+SF == 0.f) { neon_diag::TwoView().reason="ZERO_MODEL_SCORE"; return false; }')
        text=replace(text,'return ReconstructH(vbMatchesInliersH,H,', 'neon_diag::TwoView().model="H";\n            return ReconstructH(vbMatchesInliersH,H,')
        text=replace(text,'return ReconstructF(vbMatchesInliersF,F,', 'neon_diag::TwoView().model="F";\n            return ReconstructF(vbMatchesInliersF,F,')
        text=replace(text,'int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);', '''neon_diag::TwoView().inliers=N;
        neon_diag::TwoView().good=maxGood;
        const bool first=maxGood==nGood1, second=!first && maxGood==nGood2, third=!first && !second && maxGood==nGood3;
        const auto& tri=first?vbTriangulated1:(second?vbTriangulated2:(third?vbTriangulated3:vbTriangulated4));
        neon_diag::TwoView().triangulated=std::count(tri.begin(),tri.end(),true);
        neon_diag::TwoView().parallax=first?parallax1:(second?parallax2:(third?parallax3:parallax4));
        int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);''')
        text=replace(text,'''if(maxGood<nMinGood || nsimilar>1)
        {''','''neon_diag::TwoView().similar=nsimilar;
        neon_diag::TwoView().reason=maxGood<nMinGood ? "TRIANGULATION_LOW" : (nsimilar>1 ? "AMBIGUOUS_POSE" : "PARALLAX_LOW");
        if(maxGood<nMinGood || nsimilar>1)
        {''')
        text=replace(text,'if(d1/d2<1.00001 || d2/d3<1.00001)', 'neon_diag::TwoView().inliers=N;\n        neon_diag::TwoView().reason="H_DEGENERATE_SINGULAR_VALUES";\n        if(d1/d2<1.00001 || d2/d3<1.00001)')
        text=replace(text,'if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)', '''neon_diag::TwoView().good=bestGood;
        neon_diag::TwoView().triangulated=std::count(bestTriangulated.begin(),bestTriangulated.end(),true);
        neon_diag::TwoView().parallax=bestParallax;
        neon_diag::TwoView().reason=secondBestGood>=0.75*bestGood ? "AMBIGUOUS_POSE" :
            (bestParallax<minParallax ? "PARALLAX_LOW" : "TRIANGULATION_LOW");
        if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)''')
        # Only successful pose assignments in ReconstructF/H, not Find* routines.
        a=text.index('bool TwoViewReconstruction::ReconstructF(')
        b=text.index('void TwoViewReconstruction::Normalize(',a)
        part=text[a:b].replace('return true;', 'neon_diag::TwoView().reason="SUCCESS"; return true;')
        text=text[:a]+part+text[b:]
    elif name == 'Optimizer':
        text = '#include "imu_snapshot.h"\n' + text
        begin=text.index('void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg,')
        end=text.index('void Optimizer::InertialOptimization(',begin+10)
        part=text[begin:end]
        part=replace(part,'int its = 200;', 'neon_diag::ImuSnapshot diagnosticSnapshot;\n    int its = 200;')
        import re
        part=re.sub(r'optimizer.addVertex\((\w+)\);',r'optimizer.addVertex(\1); diagnosticSnapshot.Vertex(\1);',part)
        part=re.sub(r'optimizer.addEdge\((\w+)\);',r'optimizer.addEdge(\1); diagnosticSnapshot.Edge(\1);',part)
        part=replace(part,'optimizer.setVerbose(false);', 'diagnosticSnapshot.Capture(vpKFs,pMap->GetId(),maxKFid,its,priorG,priorA,bMono,bFixedVel,bGauss);\n    optimizer.setVerbose(false);')
        part=replace(part,'optimizer.optimize(its);','optimizer.optimize(its);\n    diagnosticSnapshot.Result(optimizer);')
        text=text[:begin]+part+text[end:]
        text=visual_optimizer(text)
    elif name == 'System':
        text=replace(text,'''void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;''','''void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    if(!mbResetActiveMap) neon_diag::RequestReset();
    mbResetActiveMap = true;''')
    elif name == 'Atlas':
        text=replace(text,'mspMaps.insert(mpCurrentMap);', 'mspMaps.insert(mpCurrentMap);\n    neon_diag::Event("MAP_CREATED",mpCurrentMap->GetId());')
        text=replace(text,'mpCurrentMap = pMap;', 'mpCurrentMap = pMap;\n    neon_diag::Event("MAP_CHANGED",pMap->GetId());')
    elif name == 'LocalMapping':
        begin=text.index('void LocalMapping::SearchInNeighbors()');end=text.index('\nvoid LocalMapping::',begin+10)
        part=text[begin:end]
        part=replace(part,'if (mbAbortBA)\n        return;',"""if (mbAbortBA) {
        if(neon_diag::VisualEnabled(mpCurrentKeyFrame->GetMap(),mpCurrentKeyFrame->mTimeStamp))
            neon_diag::Event("FUSION_ABORT_AFTER_FORWARD",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp);
        return;
    }
    if(neon_diag::VisualEnabled(mpCurrentKeyFrame->GetMap(),mpCurrentKeyFrame->mTimeStamp))
        neon_diag::Event("FUSION_BACKWARD_BEGIN",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp);""")
        text=text[:begin]+part+text[end:]
        begin=text.index('void LocalMapping::Run()');end=text.index('\nvoid LocalMapping::',begin+10)
        part=text[begin:end]
        for method in ['MapPointCulling','CreateNewMapPoints','SearchInNeighbors','KeyFrameCulling']:
            call=method+'();'
            before=f'neon_diag::VisualMap(mpCurrentKeyFrame->GetMap(),"{method}_before",mpCurrentKeyFrame->mTimeStamp);'
            after=f'neon_diag::VisualMap(mpCurrentKeyFrame->GetMap(),"{method}_after",mpCurrentKeyFrame->mTimeStamp);'
            part=replace(part,call,'{ '+before+' '+call+' '+after+' }')
        part=replace(part,'ProcessNewKeyFrame();', 'ProcessNewKeyFrame();\n            neon_diag::VisualMap(mpCurrentKeyFrame->GetMap(),"ProcessNewKeyFrame_after",mpCurrentKeyFrame->mTimeStamp);')
        call='Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);'
        part=replace(part,call,call+'\n                        neon_diag::VisualMap(mpCurrentKeyFrame->GetMap(),"local_ba_map_after",mpCurrentKeyFrame->mTimeStamp);')
        text=text[:begin]+part+text[end:]
        text = '#include "imu_snapshot.h"\n' + text
        text=replace(text,'mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);',
            'mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);\n            neon_diag::MapCheckpoint(mpAtlas->GetCurrentMap(),"after_scale");')
        text=replace(text,'Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);',
            'neon_diag::MapCheckpoint(mpAtlas->GetCurrentMap(),"after_full_ba");\n    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);')
        for method in ['LocalInertialBA','LocalBundleAdjustment']:
            lines=text.splitlines()
            hits=[i for i,line in enumerate(lines) if f'Optimizer::{method}(mpCurrentKeyFrame,' in line]
            if len(hits)!=1:raise ValueError(f'Pinned BA location changed: {method}')
            i=hits[0];indent=lines[i][:len(lines[i])-len(lines[i].lstrip())]
            lines[i:i+1]=[indent+'neon_diag::Event("LOCAL_BA_BEGIN",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp,"'+method+'");',
                          lines[i],indent+'neon_diag::Event("LOCAL_BA_END",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp,mbAbortBA?"ABORTED":"COMPLETED");']
            text='\n'.join(lines)+'\n'
        text=replace(text,'cout << "Not enough motion for initializing. Reseting..." << endl;', '''neon_diag::Event("LOCAL_BAD_IMU",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp,
                                    "estimated_two_baseline_distance="+std::to_string(dist),1,mpCurrentKeyFrame->GetMap()->GetIniertialBA1()?1:0,mScale);
                                cout << "Not enough motion for initializing. Reseting..." << endl;''')
        text=replace(text,'Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba,', '''neon_diag::Event("IMU_OPTIMIZATION_BEGIN",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp);
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba,''')
        text=replace(text,'if (mScale<1e-1)\n', '''neon_diag::Event("IMU_OPTIMIZATION_ESTIMATE",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp,
        mScale<1e-1 ? "SCALE_TOO_SMALL" : "ESTIMATE",mpCurrentKeyFrame->GetMap()->isImuInitialized(),
        mpCurrentKeyFrame->GetMap()->GetIniertialBA2()?2:(mpCurrentKeyFrame->GetMap()->GetIniertialBA1()?1:0),mScale,
        neon_diag::Vector(mbg.x(),mbg.y(),mbg.z()),neon_diag::Vector(mba.x(),mba.y(),mba.z()));
    if (mScale<1e-1)\n''')
        text=replace(text,'mpAtlas->SetImuInitialized();', 'mpAtlas->SetImuInitialized();\n        neon_diag::Event("IMU_INITIALIZED",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp);')
        text=replace(text,'Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);', '''Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    neon_diag::Event("SCALE_REFINEMENT_ESTIMATE",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp,"",1,2,mScale);''')
        for stage in [1,2]:
            text=replace(text,f'cout << "end VIBA {stage}" << endl;', f'neon_diag::Event("INERTIAL_BA_STAGE_{stage}_FINISHED",mpCurrentKeyFrame->GetMap()->GetId(),mpCurrentKeyFrame->mTimeStamp);\n                                cout << "end VIBA {stage}" << endl;')
    elif name == 'MapPoint':
        text='#include <tuple>\n#include <cstdlib>\n'+text
        start=text.index('void MapPoint::ComputeDistinctiveDescriptors()');end=text.index('\ncv::Mat MapPoint::GetDescriptor()',start)
        part=text[start:end]
        part=replace(part,'vector<cv::Mat> vDescriptors;','vector<cv::Mat> vDescriptors;\n    const char* mode=std::getenv("NEON_DESCRIPTOR_TIES");\n    const bool stable=mode && std::string(mode)=="source";')
        for side,index in [(0,'leftIndex'),(1,'rightIndex')]:
            old=f'vDescriptors.push_back(pKF->mDescriptors.row({index}));'
            # Preserve the upstream candidate container and its allocation in A/B.
            assert old in part
        part=replace(part,'int BestMedian = INT_MAX;',"""auto stableKey = [&observations](size_t index) {
        for(const auto& item:observations) {
            auto k=item.first;if(k->isBad())continue;
            int left=std::get<0>(item.second),right=std::get<1>(item.second);
            if(left>=0){if(index==0)return std::make_tuple(neon_diag::Source(k->mTimeStamp),0,left);--index;}
            if(right>=0){if(index==0)return std::make_tuple(neon_diag::Source(k->mTimeStamp),1,right);--index;}
        }
        throw std::runtime_error("Descriptor candidate origin not found");
    };
    int BestMedian = INT_MAX;""")
        part=replace(part,'if(median<BestMedian)','if(median<BestMedian || (stable && median==BestMedian && stableKey(i)<stableKey(BestIdx)))')
        text=text[:start]+part+text[end:]
    elif name == 'LoopClosing':
        for method in ['MergeLocal','MergeLocal2']:
            text=replace(text,f'void LoopClosing::{method}()\n{{', f'void LoopClosing::{method}()\n{{\n    neon_diag::Event("MERGE_BEGIN",mpCurrentKF->GetMap()->GetId(),mpCurrentKF->mTimeStamp,"{method}");')
        text=replace(text,'mpAtlas->SetMapBad(pCurrentMap);',
                     'mpAtlas->SetMapBad(pCurrentMap);\n        neon_diag::Event("MERGE_COMMITTED",pMergeMap->GetId(),mpCurrentKF->mTimeStamp,"from_map="+std::to_string(pCurrentMap->GetId()));')
        text=replace(text,'// Save non corrected poses (already merged maps)',
                     'neon_diag::Event("MERGE_COMMITTED",pCurrentMap->GetId(),mpCurrentKF->mTimeStamp,"from_map="+std::to_string(pMergeMap->GetId()));\n        // Save non corrected poses (already merged maps)')
    else:
        raise ValueError(name)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mapping-source',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True)
    for name in ['Tracking','TwoViewReconstruction','System','Atlas','LocalMapping','LoopClosing','Optimizer','MapPoint']:
        path=args.source/'src'/f'{name}.cc'
        if name=='LocalMapping':
            path=args.mapping_source
        generated=instrument(name,path.read_text())
        target=args.output/f'{name}_diagnostics.cc'
        if not target.exists() or target.read_text()!=generated:
            target.write_text(generated)


if __name__=='__main__':
    main()
