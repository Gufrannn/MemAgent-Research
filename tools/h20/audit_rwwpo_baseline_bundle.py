#!/usr/bin/env python3
"""Independent read-only audit of a materialized RWWPO baseline bundle."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from recurrent.research.s128_hotpot_metrics import score_terminal_output,summarize_fixed_s128
from recurrent.research.stable_eval_identity import canonical_sha256,stable_key,validate_resolved_manifest
INTERFACES=("I","Original5","Original10","Original15","Original20","Original25")
CANONICAL_AUTHORITY=(Path(__file__).resolve().parents[2]/"manifests/h20/rwwpo_original_evidence_authority_20260822.json").resolve()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reject_symlink_chain(path):
 p=Path(path)
 if any(part.is_symlink() for part in (p,*p.parents)): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:symlinked path component")
def authenticated_root(path,allowed):
 raw=Path(path)
 if any(part.is_symlink() for part in (raw,*raw.parents)): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:symlinked evidence root")
 resolved=raw.resolve(); permitted=[Path(x).resolve() for x in allowed]
 if not any(resolved==base or base in resolved.parents for base in permitted): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:unauthorized evidence root")
 return resolved
def safe_file(path,root):
 raw=Path(path); reject_symlink_chain(raw); resolved=raw.resolve(); root=Path(root).resolve()
 if not raw.is_file() or root not in resolved.parents: raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:path escape/missing")
 return resolved
def canonical_receipt(row):
 value=dict(row); declared=value.pop("report_sha256",None); return declared==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--bundle",required=True); p.add_argument("--bundle-sha256",required=True); p.add_argument("--materialization-report",required=True); p.add_argument("--final-report",required=True); p.add_argument("--resolved-manifest",required=True); p.add_argument("--authority",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--output",required=True); a=p.parse_args()
 if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=a.expected_commit: raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:checkout")
 reject_symlink_chain(a.bundle); reject_symlink_chain(a.materialization_report); reject_symlink_chain(a.final_report); reject_symlink_chain(a.resolved_manifest); reject_symlink_chain(a.authority)
 if Path(a.authority).resolve()!=CANONICAL_AUTHORITY: raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:noncanonical authority")
 bundle_path=Path(a.bundle).resolve(); root=bundle_path.parent.resolve()
 if sha(bundle_path)!=a.bundle_sha256: raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:bundle bytes")
 material=json.loads(Path(a.materialization_report).read_text()); bundle=json.loads(bundle_path.read_text()); final=json.loads(Path(a.final_report).read_text()); resolved=validate_resolved_manifest(json.loads(Path(a.resolved_manifest).read_text())); authority=json.loads(Path(a.authority).read_text()); curve=authority["original_s128_curve"]
 allowed_roots=(curve["root"],authority["original_t25_training"]["root"],authority["fixed_s128_identity"]["root"])
 if not canonical_receipt(material) or material.get("decision")!="RWWPO_BASELINE_MATERIALIZATION_PASS" or material.get("bundle_sha256")!=a.bundle_sha256 or material.get("authority_sha256")!=sha(CANONICAL_AUTHORITY): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:materialization receipt")
 if str(Path(a.final_report).resolve())!=curve["final_report"] or str(Path(a.resolved_manifest).resolve())!=curve["resolved"] or material.get("source_final_sha256")!=sha(a.final_report) or material.get("source_resolved_sha256")!=sha(a.resolved_manifest): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:source binding")
 if bundle.get("authority_sha256")!=sha(a.authority): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:authority bytes")
 if bundle.get("generator_git_commit")!=a.expected_commit or set(bundle.get("interfaces",{}))!=set(INTERFACES): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:inventory")
 for item in bundle.get("files",[]):
  source=Path(item["path"])
  reject_symlink_chain(source)
  if not source.is_file() or sha(source)!=item.get("sha256") or source.stat().st_size!=int(item.get("size",-1)): raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:source inventory bytes")
 if final.get("status")!="PASS" or final.get("evidence",{}).get("metric_rows_sha256")!=curve["canonical_metric_row_digests"]: raise SystemExit("RWWPO_BASELINE_AUDIT_NO_GO:authority final")
 identity={int(x["source_order_index"]):x for x in resolved["identity_payload"]["rows"]}; common=None; aggregates={}; row_shas={}
 for name in INTERFACES:
  item=bundle["interfaces"][name]; path=Path(item["rows_path"])
  reject_symlink_chain(path)
  if path.resolve().parent!=root or not path.is_file() or sha(path)!=item["sha256"]: raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:path/bytes")
  raw=json.loads(path.read_text()); rows=raw.get("rows") if isinstance(raw,dict) else None
  source_evidence=final["evidence"]["interfaces"][name]; source_root=authenticated_root(source_evidence["root"],allowed_roots); terminal_rel=next((x for x in source_evidence["artifacts"] if x.startswith("terminal/")),None); terminal=safe_file(source_root/terminal_rel,source_root); terminal_meta=source_evidence["artifacts"][terminal_rel]
  if terminal.is_symlink() or not terminal.is_file() or sha(terminal)!=terminal_meta["sha256"] or terminal.stat().st_size!=int(terminal_meta["size"]): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:raw terminal")
  source_rows=[json.loads(x) for x in terminal.read_text().splitlines() if x.strip()]
  if not isinstance(rows,list) or len(rows)!=128 or [x.get("source_order_index") for x in rows]!=list(range(128)): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:coverage/order")
  keys=[str(x.get("stable_key")) for x in rows]
  if len(set(keys))!=128 or (common is not None and keys!=common): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:duplicate/key drift")
  common=keys; rescored=[]
  digest_rows=[]
  for row,source_row in zip(rows,source_rows):
   order=int(row["source_order_index"]); frozen=identity[order]
   if str(row["example_id"])!=str(frozen["example_id"]) or canonical_sha256(row["ground_truth"])!=frozen["ground_truth_hash"] or row["output"]!=source_row.get("output"): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:identity/gold/output")
   score=score_terminal_output(row["output"],row["ground_truth"])
   if any(row.get(k)!=v for k,v in score.items()): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:metric tamper")
   rescored.append(score)
   digest_rows.append({"stable_key":json.dumps(stable_key(source_row),separators=(",",":")),"source_order_index":order,"eval_manifest_hash":source_row["eval_manifest_hash"],"example_id":source_row["example_id"],"replica_id":source_row["replica_id"],"trajectory_seed":source_row["trajectory_seed"],"trajectory_id":source_row["trajectory_id"],**score})
  if len(source_rows)!=128 or canonical_sha256(digest_rows)!=curve["canonical_metric_row_digests"][name] or item.get("canonical_metric_rows_sha256")!=curve["canonical_metric_row_digests"][name]: raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:canonical rows")
  aggregate=summarize_fixed_s128(rescored)
  if aggregate!=item.get("expected_aggregate"): raise SystemExit(f"RWWPO_BASELINE_AUDIT_NO_GO:{name}:aggregate")
  aggregates[name]=aggregate; row_shas[name]=sha(path)
 out={"status":"PASS","decision":"RWWPO_BASELINE_BUNDLE_AUDIT_PASS","git_commit":a.expected_commit,"authority_sha256":sha(CANONICAL_AUTHORITY),"bundle_sha256":a.bundle_sha256,"row_file_sha256":row_shas,"aggregates":aggregates}; out["report_sha256"]=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(",",":")).encode()).hexdigest(); target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
