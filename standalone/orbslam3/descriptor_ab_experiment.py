#!/usr/bin/env python3
"""Run a predeclared alternating native/source descriptor-tie comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .neon_sfm import ROOT,write_json


def run(prepared,out,trials):
    if out.exists():raise ValueError('Output must be new; preserve previous experiments')
    out.mkdir(parents=True)
    binary=ROOT/'output/orbslam3_standalone/build_diagnostics/neon_standalone'
    library=binary.parent/'libstandalone_orb.so'
    identity=dict(binary=hashlib.sha256(binary.read_bytes()).hexdigest(),library=hashlib.sha256(library.read_bytes()).hexdigest())
    plan=dict(repetitions_per_condition=trials,order='A1,B1,...',start_n=9988,max_n=65,pace=1,lost_keyframes='on',heavy_snapshots=False,light_maps=True,allocation_policy='No auxiliary candidate allocation; on-demand stable keys only for tied minima',build=identity)
    write_json(out/'plan.json',plan);rows=[]
    for i in range(1,trials+1):
        for condition,mode in [('A','native'),('B','source')]:
            if any(hashlib.sha256(path.read_bytes()).hexdigest()!=identity[key] for key,path in [('binary',binary),('library',library)]):raise RuntimeError('Binary or library changed during comparison')
            name=f'{condition}_{i:02}';target=out/name
            command=[str(ROOT/'output/orbslam3_standalone/.venv/bin/python'),str(ROOT/'standalone/orbslam3/run.py'),'--prepared',str(prepared.resolve()),'--output',str(target.resolve()),'--start_n','9988','--max_n','65','--pace','1','--lost-keyframes','on','--diagnostics','--descriptor-ties',mode,'--light-maps']
            print('START',name,flush=True)
            with (out/f'{name}.log').open('w') as log:process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(name=name,condition=condition,returncode=process.returncode)
            if process.returncode==0:
                events=[json.loads(line) for line in (target/'slam/map_events.jsonl').read_text().splitlines()];init=[e for e in events if e['event']=='IMU_INITIALIZED'];resets=[e for e in events if e['event']=='RESET_APPLIED']
                first=init[0] if init else None;followup=10052-first['source_frame_id'] if first else None
                row.update(resets=len(resets),any_reset=bool(resets),first_initialization=first,post_initialization_followup_frames=followup,right_censored_20_frames=followup is None or followup<20,pipeline=json.loads((target/'pipeline.json').read_text()))
            rows.append(row);write_json(out/'results.json',rows);print('DONE',name,flush=True)
            if process.returncode:raise RuntimeError(f'{name} failed; inspect its log, attempted runs preserved')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--trials',type=int,default=20,help='Per condition, fixed in the plan before the first run')
    args=parser.parse_args()
    if args.trials<1:parser.error('--trials must be positive')
    run(args.prepared,args.output,args.trials)
if __name__=='__main__':main()
