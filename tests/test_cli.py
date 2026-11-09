"""End-to-end CLI tests: the `aide` command as an operator/analyst uses it."""
from __future__ import annotations

import json

import pytest

from aide import cli


def _run(capsys, *argv) -> tuple[int, dict]:
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out)


@pytest.mark.parametrize("mode", ["sidecar", "blind"])
def test_embed_attack_extract_roundtrip(tmp_path, capsys, real_sample_image_path, trained_model, mode):
    stego, attacked = tmp_path / "s.png", tmp_path / "a.png"
    code, rep = _run(capsys, "embed", str(real_sample_image_path), str(stego), "--message", "grid 47N 12E",
                     "--secret", "k", "--mode", mode, "--profile", "robust")
    assert code == 0 and rep["mode"] == mode and rep["psnr_db"] > 25
    assert _run(capsys, "attack", str(stego), str(attacked), "--type", "compression",
                "--params", '{"quality": 50}')[0] == 0
    if mode == "sidecar":  # extract auto-detects the sidecar next to the *stego* path
        (tmp_path / "a.png.aide.json").write_text((tmp_path / "s.png.aide.json").read_text())
    code, rep = _run(capsys, "extract", str(attacked), "--secret", "k")
    assert code == 0 and rep["message"] == "grid 47N 12E" and rep["mode"] == mode


def test_extract_wrong_secret_exits_nonzero(tmp_path, capsys, real_sample_image_path, trained_model):
    stego = tmp_path / "s.png"
    _run(capsys, "embed", str(real_sample_image_path), str(stego), "--message", "m", "--secret", "k", "--mode", "blind")
    code, rep = _run(capsys, "extract", str(stego), "--secret", "wrong")
    assert code == 1 and rep["success"] is False


def test_secret_file(tmp_path, capsys, real_sample_image_path, trained_model):
    secret = tmp_path / "key.txt"
    secret.write_text("file-secret\n")
    stego = tmp_path / "s.png"
    _run(capsys, "embed", str(real_sample_image_path), str(stego), "--message", "m",
         "--secret-file", str(secret), "--mode", "blind")
    code, rep = _run(capsys, "extract", str(stego), "--secret", "file-secret")
    assert code == 0 and rep["message"] == "m"


def test_capacity_and_profiles(capsys, real_sample_image_path, trained_model):
    code, rep = _run(capsys, "capacity", str(real_sample_image_path))
    assert code == 0 and rep["profiles"]["quality"]["blind_max_bytes"] > rep["profiles"]["fortress"]["blind_max_bytes"]
    code, rep = _run(capsys, "profiles")
    assert code == 0 and sum(p["default"] for p in rep.values()) == 1


def test_detect_directory_screening(tmp_path, capsys, real_sample_paths):
    import shutil
    for p in real_sample_paths[:3]:
        shutil.copy(p, tmp_path / p.name)
    csv_path = tmp_path / "report.csv"
    code, rep = _run(capsys, "detect", str(tmp_path), "--csv", str(csv_path), "--all", "--no-ml")
    assert rep["summary"]["scanned"] == 3 and csv_path.exists()
    assert len(csv_path.read_text().strip().splitlines()) == 4


def test_missing_secret_is_a_usage_error(tmp_path, real_sample_image_path):
    with pytest.raises(SystemExit):
        cli.main(["embed", str(real_sample_image_path), str(tmp_path / "s.png"), "--message", "m"])
