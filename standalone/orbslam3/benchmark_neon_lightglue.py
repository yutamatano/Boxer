#!/usr/bin/env python3
"""Optional GPU-only learned comparison on the exact classical benchmark cases.

Dependencies and official weights belong in output/neon_lightglue_runtime, not
in Boxer's application environment. Reuses the identical common geometry gate.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import csv

import cv2
import numpy as np

try:
    from .benchmark_neon_correspondences import geometry, rows
except ImportError:
    from benchmark_neon_correspondences import geometry, rows

ROOT=Path(__file__).resolve().parents[2]
PINNED_LIGHTGLUE='eb42fee2d71449efb0aa5c10549752b5d75384d8'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classical',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,default=ROOT/'output/neon_lightglue_runtime')
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    sys.path[:0]=[str(args.runtime/'deps'),str(args.runtime/'source')]
    os.environ['MPLCONFIGDIR']=str(args.runtime/'mpl')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    revision=subprocess.check_output(['git','-C',str(args.runtime/'source'),'rev-parse','HEAD'],text=True).strip()
    if revision!=PINNED_LIGHTGLUE:raise ValueError('Unexpected LightGlue revision')
    if subprocess.check_output(['git','-C',str(args.runtime/'source'),'status','--porcelain'],text=True).strip():
        raise ValueError('Modified LightGlue source')
    args.output.mkdir(parents=True)
    torch.hub.set_dir(str(args.runtime/'torch_hub'))
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    from lightglue import LightGlue,SuperPoint
    import kornia
    with torch.inference_mode():
        extractor=SuperPoint(max_num_keypoints=2048).eval().cuda().float()
        # Upstream flash=True internally casts attention to FP16 even with mp=False.
        # Disable both options so this comparison's inference is explicitly FP32.
        matcher=LightGlue(features='superpoint',flash=False,mp=False).eval().cuda().float()
    if next(extractor.parameters()).device.type!='cuda' or next(matcher.parameters()).device.type!='cuda':
        raise RuntimeError('Models must remain on CUDA')
    base=rows(args.classical/'results.csv');cases=[];seen=set()
    for r in base:
        key=r['group'],int(r['reference']),int(r['target']),int(r['third']),int(r['delta'])
        if key not in seen:
            seen.add(key);cases.append(dict(zip(('group','reference','target','third','delta'),key)))
    images={}
    for line in (args.classical/'images.csv').read_text().splitlines():
        i,path=line.split(',',1);images[int(i)]=Path(path)
    fs=cv2.FileStorage(str(args.prepared/'camera.yaml'),cv2.FILE_STORAGE_READ)
    K=np.array([[fs.getNode('Camera1.fx').real(),0,fs.getNode('Camera1.cx').real()],
                [0,fs.getNode('Camera1.fy').real(),fs.getNode('Camera1.cy').real()],[0,0,1.]])
    @lru_cache(maxsize=32)
    def image(i):
        x=cv2.imread(str(images[i]),cv2.IMREAD_GRAYSCALE)
        if x is None:raise ValueError('Missing benchmark image')
        return x
    features={}
    with torch.inference_mode():
        for j,i in enumerate(sorted(images)):
            tensor=torch.from_numpy(image(i).astype(np.float32)/255.)[None].cuda()
            f=extractor.extract(tensor,resize=None)
            if f['descriptors'].dtype!=torch.float32 or f['descriptors'].device.type!='cuda':
                raise RuntimeError('Unexpected learned feature dtype/device')
            features[i]={k:v.detach().cpu() for k,v in f.items()}
            if (j+1)%100==0:print(f'GPU extracted {j+1}/{len(images)}',flush=True)
    @lru_cache(maxsize=64)
    def match(a,b):
        with torch.inference_mode():
            pair=matcher({'image0':{k:v.cuda() for k,v in features[a].items()},
                          'image1':{k:v.cuda() for k,v in features[b].items()}})
        return pair['matches'][0].detach().cpu().numpy().astype(int)
    result=[]
    for j,c in enumerate(cases):
        a,b,t=c['reference'],c['target'],c['third'];ij=match(a,b);jk=match(b,t)
        pa=features[a]['keypoints'][0].numpy()[ij[:,0]]
        pb=features[b]['keypoints'][0].numpy()[ij[:,1]]
        stats,valid=geometry(pa,pb,np.ones(len(pa)),np.ones(len(pb)),K,image(a).shape)
        continued=np.flatnonzero(np.isin(ij[:,1],jk[:,0]))
        stats['continued_three_frames']=int(np.intersect1d(valid,continued).size)
        stats['common_gate_pass']=bool(stats['essential_inliers']>=100 and stats['triangulated']>=50 and stats['grid_cells']>=6 and stats['continued_three_frames']>=50)
        result.append(dict(**c,method='superpoint_lightglue',**stats))
        if (j+1)%50==0:print(f'GPU geometry pairs {j+1}/{len(cases)}',flush=True)
    with (args.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary=[]
    for group in ('hard','good'):
        for delta in (1,2,3,5):
            rs=[r for r in result if r['group']==group and r['delta']==delta]
            summary.append(dict(group=group,delta=delta,method='superpoint_lightglue',pairs=len(rs),passes=sum(r['common_gate_pass'] for r in rs),
                                matches_median=float(np.median([r['matches'] for r in rs])),triangulated_median=float(np.median([r['triangulated'] for r in rs])),
                                continued_median=float(np.median([r['continued_three_frames'] for r in rs]))))
    weights={str(f.relative_to(args.runtime)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.runtime/'torch_hub/checkpoints').glob('*.pth')}
    identity=dict(source_url='https://github.com/cvg/LightGlue',revision=revision,torch=torch.__version__,kornia=kornia.__version__,opencv=cv2.__version__,
                  gpu=torch.cuda.get_device_name(0),neural_device='cuda',precision='float32',flash=False,mp=False,tf32=False,max_keypoints=2048,
                  weights=weights,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  classical_identity_sha256=hashlib.sha256((args.classical/'identity.json').read_bytes()).hexdigest(),feature_counts={i:f['keypoints'].shape[1] for i,f in features.items()})
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(summary=summary,limitation='Same essential/triangulation/continuation proxy as classical benchmark; not trajectory coverage or absolute accuracy.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
