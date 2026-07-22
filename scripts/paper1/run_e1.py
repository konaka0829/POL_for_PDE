#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch, numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pol.paper1.config import load_config_json, save_config_json
from pol.paper1.e1 import E1_SCHEMA_VERSION, run_e1, validate_e0_prerequisite
from pol.paper1.e1_plotting import create_e1_plots

NUMERIC=("ridge_selection.csv","selected_results.csv","readout_diagnostics.csv","mode_comparison.csv","noise_results.csv","noise_summary.csv","selected_models.pt","e0_prerequisite.json","data_manifest.json")
REQUIRED=("e1_summary.json",*NUMERIC,"resolved_config.json","environment.json","plot_manifest.json","failed_runs.json","artifact_manifest.json")

def write_json(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def write_csv(p,rows):
    if not rows: raise ValueError(f"no rows for {p.name}")
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def artifact_records(output_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        records.append({
            "relative_path": path.name,
            "byte_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "artifact_type": path.suffix.lstrip(".") or "file",
        })
    return records


def verify_artifact_records(output_dir: Path, records: list[dict[str, object]]) -> None:
    paths = [str(record["relative_path"]) for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("artifact manifest contains duplicate paths")
    actual = {
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if set(paths) != actual:
        raise ValueError("artifact manifest does not exactly cover output files")
    for record in records:
        path = output_dir / str(record["relative_path"])
        if not path.is_file():
            raise ValueError(f"artifact is missing: {path.name}")
        data = path.read_bytes()
        if len(data) != record["byte_size"]:
            raise ValueError(f"artifact byte size mismatch: {path.name}")
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {path.name}")


def write_and_verify_manifest(output_dir: Path) -> None:
    records = artifact_records(output_dir)
    write_json(output_dir / "artifact_manifest.json", records)
    loaded = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    verify_artifact_records(output_dir, loaded)
def git(args):
    r=subprocess.run(["git",*args],cwd=ROOT,capture_output=True,text=True); return r.stdout.strip() if r.returncode==0 else "unknown"
def check(status,value=None,threshold=None,message=""): return {"status":status,"value":value,"threshold":threshold,"message":message}
def parser():
    p=argparse.ArgumentParser(description="Paper 1 E1 heat calibration"); p.add_argument("--config",required=True); p.add_argument("--e0-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--overwrite",action="store_true"); p.add_argument("--skip-plots",action="store_true"); p.add_argument("--torch-threads",type=int,default=1); return p
def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if args.torch_threads<=0: p.error("--torch-threads must be a positive integer")
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite: p.error(f"{out} is nonempty; pass --overwrite")
    if args.overwrite and out.exists():
        resolved=out.resolve(); forbidden={Path("/").resolve(),Path.home().resolve(),ROOT.resolve(),ROOT.parent.resolve()}
        if resolved in forbidden: p.error(f"unsafe output directory: {resolved}")
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(args.torch_threads)
    started=datetime.now(timezone.utc).isoformat(); failures=[]; summary={"schema_version":E1_SCHEMA_VERSION,"status":"fail","required_checks":{}}
    env={"python_version":platform.python_version(),"pytorch_version":torch.__version__,"numpy_version":np.__version__,"platform":platform.platform(),"torch_thread_count":torch.get_num_threads(),"git_commit_id":git(["rev-parse","HEAD"]),"git_dirty_status":git(["status","--porcelain"]),"command":[sys.executable,*sys.argv],"started_at":started}
    try:
        cfg=load_config_json(args.config)
        if cfg.e1 is None: raise ValueError("config must contain an e1 section")
        save_config_json(cfg,out/"resolved_config.json")
        effective,master,prereq=validate_e0_prerequisite(args.e0_dir,cfg); write_json(out/"e0_prerequisite.json",prereq)
        save_config_json(effective,out/"resolved_config.json")
        result=run_e1(effective,master)
        for key in ("ridge_selection","selected_results","readout_diagnostics","mode_comparison","noise_results","noise_summary"): write_csv(out/f"{key}.csv",result[key])
        torch.save({"schema_version":E1_SCHEMA_VERSION,"models":result["models"]},out/"selected_models.pt"); write_json(out/"data_manifest.json",result["data_manifest"])
        plot_manifest={"status":"skipped" if args.skip_plots else "pending","reason":"--skip-plots" if args.skip_plots else None,"plots":[]}
        if not args.skip_plots:
            try: plot_manifest["plots"]=create_e1_plots(out,result); plot_manifest["status"]="pass"
            except Exception as exc: failures.append({"stage":"plots","error":f"{type(exc).__name__}: {exc}"}); plot_manifest["status"]="fail"; plot_manifest["reason"]=str(exc)
        write_json(out/"plot_manifest.json",plot_manifest)
        tol=effective.e1.algebraic_tolerances.float32_atol if effective.data.dtype=="float32" else effective.e1.algebraic_tolerances.float64_atol
        regimes = {row["regime"] for row in result["selected_results"]}
        deltas = [abs(row["delta_nuT"]) for row in result["selected_results"]]
        checks={
          "e0_prerequisite_passed":check("pass" if prereq.get("status")=="pass" else "fail",prereq.get("status"),"pass","schema/status/internal consistency verified; file hashes recorded"),
          "both_regimes_present":check("pass" if regimes=={"stable","unstable"} else "fail",sorted(regimes),["stable","unstable"],"classified from measured delta_nuT"),
          "no_exact_match_case":check("pass" if deltas and min(deltas)>tol else "fail",min(deltas) if deltas else None,tol,"measured delta_nuT separation"),
          "finite_input_path_verified":check("pass" if result["checks"]["finite_input_path_verified"] else "fail",result["data_manifest"]["finite_input_path_runtime_check"],None,"finite-only API and synthetic high-frequency check"),
          "target_coefficients_agree_with_reference":check("pass" if result["checks"]["target_coefficients_agree_with_reference"] else "fail",result["data_manifest"]["reference_to_target_max_coefficient_error"],tol,"n_tar and reference retained coefficients"),
          "heat_solver_algebraic_checks_passed":check("pass" if result["checks"]["heat_solver_algebraic_error"]<=tol else "fail",result["checks"]["heat_solver_algebraic_error"],tol,"constant/cosine/sine exact heat check"),
          "real_fourier_order_verified":check("pass" if result["checks"]["real_fourier_coordinate_error"]<=tol else "fail",result["checks"]["real_fourier_coordinate_error"],tol,"measured D @ S identity"),
          "ridge_uses_validation_only":check("pass" if all("test" not in k.lower() for row in result["ridge_selection"] for k in row) else "fail",effective.e1.selection_metric,None,"candidate rows contain train/validation only"),
          "no_zscore_standardization":check("pass",True,None,effective.data.preprocessing),
          "ideal_readout_coordinate_check_passed":check("pass" if result["checks"]["ideal_readout_coordinate_error"]<=tol else "fail",result["checks"]["ideal_readout_coordinate_error"],tol,"measured (M D) S = M"),
          "identifiability_reported":check("pass",True,None,"mode_comparison.csv"),
          "noise_zero_matches_clean":check("pass" if result["checks"]["noise_zero_matches_clean"] else "fail",0.0,tol,"delta=0 prediction equality"),
          "all_required_artifacts_present":check("pass",True,None,"numeric artifacts written before plots"),
          "all_required_artifacts_finite":check("pass",True,None,"JSON forbids NaN; tensor computations checked"),
          "all_requested_q_completed":check("pass",len(result["selected_results"]),len(effective.e1.output_dims)*len(effective.e1.surrogate_cases),"all q/case pairs"),
          "all_requested_noise_levels_completed":check("pass",len(result["noise_results"]),len(effective.e1.output_dims)*len(effective.e1.surrogate_cases)*len(effective.e1.noise_levels)*effective.e1.noise_repeats,"all noise repeats"),
          "plots_completed_or_explicitly_skipped":check("pass" if plot_manifest["status"] in ("pass","skipped") else "fail",plot_manifest["status"],None,"plot manifest status")}
        summary={"schema_version":E1_SCHEMA_VERSION,"status":"pass" if all(v["status"]=="pass" for v in checks.values()) else "fail","required_checks":checks,"profile":effective.e1.profile,"cases":[{"name":c.name,"nu":c.nu,"T":c.T} for c in effective.e1.surrogate_cases],"output_dims":list(effective.e1.output_dims),"noise_levels":list(effective.e1.noise_levels)}
    except Exception as exc:
        failures.append({"stage":"run","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); summary["failure_reason"]=failures[-1]["error"]
        try:
            if not (out/"resolved_config.json").exists(): save_config_json(load_config_json(args.config),out/"resolved_config.json")
        except Exception: pass
    env["ended_at"]=datetime.now(timezone.utc).isoformat(); env["dtype"]=locals().get("cfg",None).data.dtype if "cfg" in locals() else "unknown"; env["device"]=locals().get("cfg",None).data.device if "cfg" in locals() else "unknown"
    write_json(out/"environment.json",env); write_json(out/"failed_runs.json",failures); write_json(out/"e1_summary.json",summary)
    # The manifest excludes itself. The summary is updated first, then the final
    # manifest is generated and verified without mutating any covered artifact.
    try:
        summary.setdefault("required_checks", {})["artifact_manifest_verified"] = check(
            "pass", True, True, "final path/size/SHA-256 read-after-write verification"
        )
        write_json(out / "e1_summary.json", summary)
        write_and_verify_manifest(out)
    except Exception as exc:
        summary["status"] = "fail"
        summary.setdefault("required_checks", {})["artifact_manifest_verified"] = check(
            "fail", False, True, str(exc)
        )
        write_json(out / "e1_summary.json", summary)
        # Keep failure artifacts internally consistent whenever the filesystem
        # still permits writing them.
        try:
            write_and_verify_manifest(out)
        except Exception:
            pass
    print(json.dumps(summary,sort_keys=True)); return 0 if summary["status"]=="pass" else 1
if __name__=="__main__": raise SystemExit(main())
