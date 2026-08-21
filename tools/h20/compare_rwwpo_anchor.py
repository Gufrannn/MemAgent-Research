#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--method",required=True); p.add_argument("--baseline-import",required=True); p.add_argument("--step",type=int,choices=[5,10,15,20,25],required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--output",required=True); a=p.parse_args()
 method=json.loads(Path(a.method).read_text()); base=json.loads(Path(a.baseline_import).read_text()); key=f"Original{a.step}"
 if method.get("decision")!=f"RWWPO_T{a.step}_S128_PASS" or key not in base.get("aggregates",{}): raise SystemExit("RWWPO_COMPARE_NO_GO:missing certified inputs")
 delta=method["metrics"]["token_f1"]-base["aggregates"][key]["token_f1"]; passed=delta>=-.02
 report={"status":"PASS" if passed else "FAIL","decision":f"RWWPO_T{a.step}_COMPARE_PASS" if passed else f"RWWPO_T{a.step}_COMPARE_NO_GO","git_commit":a.expected_commit,"step":a.step,"token_f1_delta":delta,"method_sha256":hashlib.sha256(Path(a.method).read_bytes()).hexdigest(),"baseline_import_sha256":hashlib.sha256(Path(a.baseline_import).read_bytes()).hexdigest()}
 raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest(); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n"); raise SystemExit(0 if passed else 1)
if __name__=="__main__": main()
