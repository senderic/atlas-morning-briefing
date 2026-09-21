"""Behavior checks for the sequential cron wrapper."""

import os
import shutil
import subprocess
import time


def test_sequential_wrapper_exports_one_codex_deadline_to_both_runners(tmp_path):
    """Catches main and local runs receiving separate Codex timeout windows."""
    root = tmp_path / "briefing"
    root.mkdir()
    wrapper = root / "run_briefing.sh"
    shutil.copy2("run_briefing.sh", wrapper)
    python = root / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    observed = tmp_path / "deadlines.txt"
    python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$ATLAS_CODEX_DEADLINE_EPOCH" >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
    )
    python.chmod(0o755)

    started = time.time()
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=root,
        env={
            **os.environ,
            "ATLAS_WRAPPER_OBSERVATIONS": str(observed),
            "SKIP_QUALITY_CHECK": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    deadlines = observed.read_text().splitlines()
    assert len(deadlines) == 3  # preflight, Atlas, then local runner
    assert len(set(deadlines)) == 1
    assert started + 295 <= float(deadlines[0]) <= time.time() + 305
