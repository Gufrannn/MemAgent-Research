#!/usr/bin/env python3
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss, compute_rwwpo_policy_loss


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--expected-commit", required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    if head != a.expected_commit: raise SystemExit("RWWPO_E0_NO_GO: commit mismatch")
    old=torch.zeros((4,3),dtype=torch.float64); mask=torch.tensor([[1,1,0],[1,0,0],[1,1,1],[1,1,1]],dtype=torch.bool)
    final=torch.tensor([0,0,1,1],dtype=torch.bool); writer=mask & (~final).unsqueeze(-1)
    sid=torch.tensor([0,1,0,1]); turn=torch.tensor([0,0,1,1]); scalar=torch.tensor([.7,-.4,.7,-.4],dtype=torch.float64)
    adv=scalar[:,None].expand_as(old)*mask
    x=old.clone().requires_grad_(True); lo,*_=compute_policy_loss(old,x,adv,mask,.2,.2,.2,loss_agg_mode="token-mean"); go,=torch.autograd.grad(lo,x)
    y=old.clone().requires_grad_(True); lr,m=compute_rwwpo_policy_loss(old,y,adv,mask,writer,final,sid,turn,.2,.2,.2); gr,=torch.autograd.grad(lr,y)
    error=float((go-gr).abs().max()); closure=bool(torch.equal(mask,writer|m["answer_mask"]) and not (writer&m["answer_mask"]).any())
    cosine=float(torch.nn.functional.cosine_similarity(go.flatten(),gr.flatten(),dim=0))
    direction=torch.arange(old.numel(),dtype=old.dtype).reshape_as(old); direction=direction/direction.norm(); eps=1e-6
    def directional(which,sign):
        value=old+sign*eps*direction
        if which=="original": return compute_policy_loss(old,value,adv,mask,.2,.2,.2,loss_agg_mode="token-mean")[0]
        return compute_rwwpo_policy_loss(old,value,adv,mask,writer,final,sid,turn,.2,.2,.2)[0]
    fd_original=float((directional("original",1)-directional("original",-1))/(2*eps)); fd_rwwpo=float((directional("rwwpo",1)-directional("rwwpo",-1))/(2*eps)); fd_error=abs(fd_original-fd_rwwpo)
    config=Path("verl/trainer/config/ppo_trainer.yaml").read_text(); off_default="rwwpo:\n      enable: false" in config
    status="PASS" if error <= 1e-12 and fd_error<=1e-9 and cosine>1-1e-12 and closure and off_default else "FAIL"
    report={"status":status,"decision":"RWWPO_E0_PASS" if status=="PASS" else "RWWPO_E0_NO_GO","git_commit":head,
            "max_abs_gradient_error":error,"gradient_cosine":cosine,"finite_difference_error":fd_error,"mask_closure":closure,"original_off_default":off_default}
    raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status=="PASS" else 1)
if __name__=="__main__": main()
