#!/usr/bin/env python3
"""Fail-closed final certificate over all RWWPO anchors."""
import argparse, hashlib, json, subprocess
from pathlib import Path
def verified(path,commit):
 row=json.loads(Path(path).read_text()); declared=row.pop("report_sha256",None); actual=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 if declared!=actual or row.get("status")!="PASS" or row.get("git_commit")!=commit: raise SystemExit(f"RWWPO_FINAL_NO_GO:invalid receipt {path}")
 return row,declared
def main():
 p=argparse.ArgumentParser(); p.add_argument("--certificate-root",required=True); p.add_argument("--baseline-import",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--output",required=True); a=p.parse_args(); root=Path(a.certificate_root); anchors=[]
 if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=a.expected_commit: raise SystemExit("RWWPO_FINAL_NO_GO:checkout commit")
 baseline,bsha=verified(a.baseline_import,a.expected_commit)
 if baseline.get("decision")!="ORIGINAL_BASELINE_IMPORT_PASS": raise SystemExit("RWWPO_FINAL_NO_GO:baseline decision")
 baseline_file_sha=hashlib.sha256(Path(a.baseline_import).read_bytes()).hexdigest()
 for step in (5,10,15,20,25):
  health,hsha=verified(root/f"t{step}_health.json",a.expected_commit); method,msha=verified(root/f"t{step}_s128.json",a.expected_commit); compare,csha=verified(root/f"t{step}_compare.json",a.expected_commit)
  if health.get("decision")!=f"RWWPO_T{step}_HEALTH_PASS" or method.get("decision")!=f"RWWPO_T{step}_S128_PASS" or compare.get("decision")!=f"RWWPO_T{step}_COMPARE_PASS": raise SystemExit(f"RWWPO_FINAL_NO_GO:wrong decision T{step}")
  if compare.get("method_sha256")!=hashlib.sha256((root/f"t{step}_s128.json").read_bytes()).hexdigest() or compare.get("baseline_import_sha256")!=baseline_file_sha: raise SystemExit(f"RWWPO_FINAL_NO_GO:comparison binding T{step}")
  delta=float(method["metrics"]["token_f1"])-float(baseline["aggregates"][f"Original{step}"]["token_f1"])
  if abs(delta-float(compare.get("token_f1_delta",float("nan"))))>1e-12: raise SystemExit(f"RWWPO_FINAL_NO_GO:comparison delta mismatch T{step}")
  anchors.append({"step":step,"token_f1":method["metrics"]["token_f1"],"delta":delta,"health_sha256":hsha,"method_sha256":msha,"compare_sha256":csha})
 mean_delta=sum(x["delta"] for x in anchors)/5; passed=anchors[-1]["delta"]>=.02 and mean_delta>=.01 and min(x["delta"] for x in anchors)>=-.02
 report={"status":"PASS" if passed else "FAIL","decision":"RWWPO_FIVE_ANCHOR_PASS" if passed else "RWWPO_FIVE_ANCHOR_NO_GO","git_commit":a.expected_commit,"anchors":anchors,"mean_token_f1_delta":mean_delta}; report["report_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n"); raise SystemExit(0 if passed else 1)
if __name__=="__main__": main()
