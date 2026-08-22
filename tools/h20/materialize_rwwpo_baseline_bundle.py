#!/usr/bin/env python3
"""Materialize an RWWPO baseline bundle only from authenticated raw S128 artifacts."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from recurrent.research.s128_hotpot_metrics import score_terminal_output,summarize_fixed_s128
from recurrent.research.stable_eval_identity import canonical_sha256,stable_key,validate_resolved_manifest
from tools.h20.audit_qwen25_7b_s128_it import _ground_truth_by_source_order

INTERFACES=("I","Original5","Original10","Original15","Original20","Original25")
CANONICAL_AUTHORITY=(Path(__file__).resolve().parents[2]/"manifests/h20/rwwpo_original_evidence_authority_20260822.json").resolve()
FORBIDDEN_SOURCE_GOLD={"ground_truth","gold","gold_answer","reference_answer","references"}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def reject_symlink_chain(path):
 p=Path(path)
 if any(part.is_symlink() for part in (p,*p.parents)): raise ValueError(f"symlinked path component: {p}")
def authenticated_root(path,allowed):
 raw=Path(path)
 if any(part.is_symlink() for part in (raw,*raw.parents)): raise ValueError(f"symlinked evidence root: {raw}")
 resolved=raw.resolve(); permitted=[Path(x).resolve() for x in allowed]
 if not any(resolved==base or base in resolved.parents for base in permitted): raise ValueError(f"unauthorized evidence root: {raw}")
 return resolved
def safe_file(path,root):
 p=Path(path); raw_root=Path(root); root=raw_root.resolve()
 cursor=p
 symlinked=raw_root.is_symlink()
 while cursor!=raw_root and cursor!=cursor.parent:
  symlinked=symlinked or cursor.is_symlink(); cursor=cursor.parent
 if symlinked or not p.is_file() or root not in p.resolve().parents: raise ValueError(f"path escape/symlink/missing: {p}")
 return p.resolve()
def expected_file(path,digest,size=None):
 p=Path(path)
 reject_symlink_chain(p)
 if not p.is_file() or sha(p)!=digest or (size is not None and p.stat().st_size!=int(size)): raise ValueError(f"authenticated file mismatch: {p}")
 return p.resolve()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--final-report",required=True); p.add_argument("--final-report-sha256",required=True); p.add_argument("--resolved-manifest",required=True); p.add_argument("--resolved-sha256",required=True); p.add_argument("--execution-ledger",required=True); p.add_argument("--ledger-sha256",required=True); p.add_argument("--ledger-tail-sha256",required=True); p.add_argument("--validation",required=True); p.add_argument("--validation-sha256",required=True); p.add_argument("--authority",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--output-root",required=True); a=p.parse_args()
 head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 if head!=a.expected_commit or subprocess.check_output(["git","status","--porcelain"],text=True).strip(): raise SystemExit("RWWPO_BASELINE_NO_GO:checkout")
 final=expected_file(a.final_report,a.final_report_sha256); resolved_path=expected_file(a.resolved_manifest,a.resolved_sha256); ledger=expected_file(a.execution_ledger,a.ledger_sha256); validation=expected_file(a.validation,a.validation_sha256); authority_path=Path(a.authority).resolve()
 if authority_path!=CANONICAL_AUTHORITY: raise SystemExit("RWWPO_BASELINE_NO_GO:noncanonical authority")
 ledger_lines=[x for x in ledger.read_bytes().splitlines(keepends=True) if x.strip()]
 if not ledger_lines or hashlib.sha256(ledger_lines[-1]).hexdigest()!=a.ledger_tail_sha256: raise SystemExit("RWWPO_BASELINE_NO_GO:ledger tail")
 authority=json.loads(authority_path.read_text()); curve=authority["original_s128_curve"]
 interface_roots=curve["authenticated_interface_roots"]
 if set(interface_roots)!=set(INTERFACES): raise SystemExit("RWWPO_BASELINE_NO_GO:authority interface roots")
 if str(final)!=curve["final_report"] or str(resolved_path)!=curve["resolved"] or str(ledger)!=curve["ledger"]: raise SystemExit("RWWPO_BASELINE_NO_GO:authority path drift")
 report=json.loads(final.read_text()); resolved=validate_resolved_manifest(json.loads(resolved_path.read_text()))
 if report.get("status")!="PASS" or report.get("decision")!="ORIGINAL_S128_CURVE_PASS": raise SystemExit("RWWPO_BASELINE_NO_GO:source final report")
 if report.get("evidence",{}).get("metric_rows_sha256")!=curve["canonical_metric_row_digests"]: raise SystemExit("RWWPO_BASELINE_NO_GO:canonical row digest authority")
 truth=_ground_truth_by_source_order({"data":{"validation":str(validation)}},resolved)
 out=Path(a.output_root)
 if out.exists(): raise SystemExit("RWWPO_BASELINE_NO_GO:output root already exists")
 out.mkdir(parents=True); (out/"RUN_ID_CONSUMED").write_text(f"commit={head}\ncreated={int(time.time())}\n")
 interfaces={}; files=[]; key_inventory=None
 for name in INTERFACES:
  evidence=report["evidence"]["interfaces"][name]; source_root=authenticated_root(evidence["root"],[interface_roots[name]])
  if source_root!=Path(interface_roots[name]).resolve(): raise ValueError(f"{name}:interface root drift")
  terminal_rel=next((x for x in evidence["artifacts"] if x.startswith("terminal/")),None)
  if terminal_rel is None: raise ValueError(f"{name}:missing terminal inventory")
  authenticated={}
  for rel,artifact in evidence["artifacts"].items():
   source=expected_file(safe_file(source_root/rel,source_root),artifact["sha256"],artifact["size"]); authenticated[rel]=source; files.append({"path":str(source),"sha256":sha(source),"size":source.stat().st_size})
  meta=evidence["artifacts"][terminal_rel]; terminal=expected_file(safe_file(source_root/terminal_rel,source_root),meta["sha256"],meta["size"])
  source_rows=[json.loads(x) for x in terminal.read_text().splitlines() if x.strip()]
  if len(source_rows)!=128 or [int(x["source_order_index"]) for x in source_rows]!=list(range(128)): raise ValueError(f"{name}:coverage/order")
  if any(FORBIDDEN_SOURCE_GOLD & set(row) for row in source_rows): raise ValueError(f"{name}:gold leaked into terminal artifact")
  rows=[]
  for row in source_rows:
   order=int(row["source_order_index"]); scored=score_terminal_output(row["output"],truth[order]); rows.append({"stable_key":json.dumps(stable_key(row),separators=(",",":")),"source_order_index":order,"example_id":str(row["example_id"]),"output":row["output"],"ground_truth":truth[order],**scored})
  digest_rows=[{"stable_key":json.dumps(stable_key(src),separators=(",",":")),"source_order_index":int(src["source_order_index"]),"eval_manifest_hash":src["eval_manifest_hash"],"example_id":src["example_id"],"replica_id":src["replica_id"],"trajectory_seed":src["trajectory_seed"],"trajectory_id":src["trajectory_id"],**score_terminal_output(src["output"],truth[int(src["source_order_index"])])} for src in source_rows]
  if canonical_sha256(digest_rows)!=curve["canonical_metric_row_digests"][name]: raise ValueError(f"{name}:independent metric rows digest")
  keys=[x["stable_key"] for x in rows]
  if len(set(keys))!=128 or (key_inventory is not None and keys!=key_inventory): raise ValueError(f"{name}:stable key drift")
  key_inventory=keys; rows_path=out/f"{name}.rows.json"; rows_path.write_text(json.dumps({"rows":rows},sort_keys=True,separators=(",",":"))+"\n")
  interfaces[name]={"rows_path":str(rows_path.resolve()),"sha256":sha(rows_path),"canonical_metric_rows_sha256":curve["canonical_metric_row_digests"][name],"expected_aggregate":summarize_fixed_s128(rows)}
 for source in (final,resolved_path,ledger,validation): files.append({"path":str(source),"sha256":sha(source),"size":source.stat().st_size})
 bundle={"schema_version":"rwwpo-baseline-bundle-v1","generator_git_commit":head,"authority_path":str(authority_path),"authority_sha256":sha(authority_path),"files":files,"interfaces":interfaces}
 bundle_path=out/"baseline_bundle.json"; bundle_path.write_text(json.dumps(bundle,sort_keys=True,indent=2)+"\n")
 receipt={"status":"PASS","decision":"RWWPO_BASELINE_MATERIALIZATION_PASS","git_commit":head,"authority_sha256":sha(authority_path),"bundle_path":str(bundle_path.resolve()),"bundle_sha256":sha(bundle_path),"source_final_sha256":sha(final),"source_resolved_sha256":sha(resolved_path),"source_ledger_sha256":sha(ledger),"source_ledger_tail_sha256":a.ledger_tail_sha256,"row_file_sha256":{k:v["sha256"] for k,v in interfaces.items()}}; receipt["report_sha256"]=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()).hexdigest(); (out/"materialization_report.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
 print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__": main()
