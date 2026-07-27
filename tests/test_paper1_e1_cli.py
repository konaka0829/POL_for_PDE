import json
import subprocess
import sys

import pytest

def test_cli_rejects_stale_png_without_overwrite(tmp_path):
    out=tmp_path/"out"; out.mkdir(); (out/"old.png").write_bytes(b"stale")
    p=subprocess.run([sys.executable,"scripts/paper1/run_e1.py","--config","configs/paper1_e1_smoke.json","--e0-dir",str(tmp_path/"missing"),"--output-dir",str(out)],capture_output=True,text=True)
    assert p.returncode!=0 and "nonempty" in p.stderr

@pytest.fixture
def e0_dir(tmp_path):
    out=tmp_path/"e0"
    process=subprocess.run([sys.executable,"scripts/paper1/run_e0.py","--config","configs/paper1_e0_for_e1_smoke.json","--output-dir",str(out),"--overwrite"],capture_output=True,text=True)
    assert process.returncode==0,process.stdout+process.stderr
    return out

@pytest.mark.parametrize("mutation,reason",[
    (lambda raw: raw["e1"].__setitem__("min_identifiable_nonconstant_fraction",0.99),"scientific"),
    (lambda raw: raw["e1"].__setitem__("noise_levels",[0.0,1e308]),"non-finite"),
])
def test_cli_gate_and_saved_qa_fail_nonzero(e0_dir,tmp_path,mutation,reason):
    raw=json.load(open("configs/paper1_e1_smoke.json")); mutation(raw)
    config=tmp_path/(reason+".json"); config.write_text(json.dumps(raw))
    out=tmp_path/(reason+"-out")
    process=subprocess.run([sys.executable,"scripts/paper1/run_e1.py","--config",str(config),"--e0-dir",str(e0_dir),"--output-dir",str(out),"--overwrite","--skip-plots","--torch-threads","1"],capture_output=True,text=True)
    assert process.returncode!=0
    summary=json.loads((out/"e1_summary.json").read_text())
    assert summary["status"]=="fail" and reason in summary["failure_reason"].lower()
    assert {path.name for path in out.iterdir()} == {
        "e1_summary.json", "environment.json", "failed_runs.json"
    }
