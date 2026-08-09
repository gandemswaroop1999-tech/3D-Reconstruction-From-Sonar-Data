"""Render one PCN checkpoint against a fixed partial/ground-truth sample."""
import argparse
from pathlib import Path

import sys

# The PCN architecture lives in the sibling 02_pcn_training folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_pcn_training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_from_dataset import PCN


def plot_points(axis, points, title, color):
    center=points.mean(axis=0); radius=max(np.ptp(points,axis=0).max()/2,1e-3)
    axis.scatter(points[:,0],points[:,1],points[:,2],s=0.55,c=color,alpha=0.78,linewidths=0)
    axis.set_xlim(center[0]-radius,center[0]+radius); axis.set_ylim(center[1]-radius,center[1]+radius); axis.set_zlim(center[2]-radius,center[2]+radius)
    axis.set_box_aspect((1,1,1)); axis.view_init(elev=18,azim=-58); axis.set_title(title,color="#f8fafc",fontsize=14,fontweight="bold",pad=12)
    axis.set_axis_off(); axis.set_facecolor("#0f172a")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True); parser.add_argument("--data",required=True)
    parser.add_argument("--epoch",required=True,type=int); parser.add_argument("--out",required=True); parser.add_argument("--sample-index",type=int,default=42)
    parser.add_argument("--object-name",default="SUOP object")
    args=parser.parse_args()
    dataset=np.load(args.data); partial=dataset["parts"][args.sample_index]; ground_truth=dataset["gts"][args.sample_index]
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=PCN().eval(); model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad(): _,prediction=model(torch.from_numpy(partial).float().unsqueeze(0))
    figure=plt.figure(figsize=(15.6,5.8),facecolor="#0f172a")
    figure.suptitle(f"SUOP {args.object_name.title()} PCN Completion — Epoch {args.epoch}",color="white",fontsize=20,fontweight="bold",y=0.93)
    for index,(label,points,color) in enumerate((("Input sonar partial",partial,"#38bdf8"),(f"SUOP {args.object_name.lower()} reference surface",ground_truth,"#cbd5e1"),("PCN completion",prediction.squeeze(0).numpy(),"#f97316")),start=1):
        plot_points(figure.add_subplot(1,3,index,projection="3d"),points,label,color)
    figure.subplots_adjust(left=0.02,right=0.98,bottom=0.05,top=0.80,wspace=0.03)
    output=Path(args.out); output.parent.mkdir(parents=True,exist_ok=True); figure.savefig(output,dpi=180,facecolor=figure.get_facecolor())
    print(output)


if __name__=="__main__": main()
