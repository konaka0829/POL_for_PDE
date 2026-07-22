import subprocess,sys

def test_cli_rejects_stale_png_without_overwrite(tmp_path):
    out=tmp_path/"out"; out.mkdir(); (out/"old.png").write_bytes(b"stale")
    p=subprocess.run([sys.executable,"scripts/paper1/run_e1.py","--config","configs/paper1_e1_smoke.json","--e0-dir",str(tmp_path/"missing"),"--output-dir",str(out)],capture_output=True,text=True)
    assert p.returncode!=0 and "nonempty" in p.stderr

