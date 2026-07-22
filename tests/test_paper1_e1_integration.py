import csv,json,subprocess,sys
import pytest

@pytest.mark.parametrize("skip_plots",[True,False])
def test_e0_e1_smoke_validation_only(tmp_path,skip_plots):
    e0,e1=tmp_path/"e0",tmp_path/"e1"
    a=subprocess.run([sys.executable,"scripts/paper1/run_e0.py","--config","configs/paper1_e0_for_e1_smoke.json","--output-dir",str(e0),"--overwrite"],capture_output=True,text=True)
    assert a.returncode==0,a.stdout+a.stderr
    command=[sys.executable,"scripts/paper1/run_e1.py","--config","configs/paper1_e1_smoke.json","--e0-dir",str(e0),"--output-dir",str(e1),"--overwrite","--torch-threads","1"]
    if skip_plots: command.append("--skip-plots")
    b=subprocess.run(command,capture_output=True,text=True)
    assert b.returncode==0,b.stdout+b.stderr
    assert json.loads((e1/"e1_summary.json").read_text())["status"]=="pass"
    with (e1/"ridge_selection.csv").open() as f: fields=next(csv.reader(f))
    assert not any("test" in x.lower() for x in fields)
    manifest=json.loads((e1/"plot_manifest.json").read_text())
    assert manifest["status"]==("skipped" if skip_plots else "pass")
    assert not list(e1.glob("*.png")) if skip_plots else all((e1/r["relative_path"]).is_file() for r in manifest["plots"])
