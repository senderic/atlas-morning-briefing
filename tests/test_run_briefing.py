"""Behavior checks for the sequential cron wrapper."""

import os
import shutil
import subprocess


def test_sequential_wrapper_does_not_export_a_codex_wall_clock_deadline(tmp_path):
    """Catches pipeline startup time suppressing later report-writer calls."""
    root = tmp_path / "briefing"
    root.mkdir()
    wrapper = root / "run_briefing.sh"
    shutil.copy2("run_briefing.sh", wrapper)
    python = root / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    observed = tmp_path / "deadline-presence.txt"
    python.write_text(
        "#!/bin/bash\n"
        'if [ -n "${ATLAS_CODEX_DEADLINE_EPOCH+x}" ]; then '
        'printf "set\\n"; else printf "unset\\n"; fi >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
    )
    python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=root,
        env={
            **{k: v for k, v in os.environ.items() if k != "ATLAS_CODEX_DEADLINE_EPOCH"},
            "ATLAS_WRAPPER_OBSERVATIONS": str(observed),
            "SKIP_QUALITY_CHECK": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    observations = observed.read_text().splitlines()
    assert observations == ["unset", "unset", "unset"]
