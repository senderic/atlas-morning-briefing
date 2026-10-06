"""Behavior checks for the sequential cron wrapper."""

import os
import shutil
import subprocess


def _run_mode(tmp_path, mode_env):
    root = tmp_path / "briefing"
    root.mkdir()
    wrapper = root / "run_briefing.sh"
    shutil.copy2("run_briefing.sh", wrapper)

    python = root / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    observed = tmp_path / "wrapper-actions.txt"
    python.write_text(
        "#!/bin/bash\n"
        'printf "python:%s\\n" "$*" >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
    )
    python.chmod(0o755)

    quality = root / "run_quality_check.sh"
    quality.write_text(
        "#!/bin/bash\n"
        'printf "quality:%s\\n" "$*" >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
    )
    quality.chmod(0o755)

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=root,
        env={
            **os.environ,
            **mode_env,
            "ATLAS_WRAPPER_OBSERVATIONS": str(observed),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    return observed.read_text().splitlines()


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


def test_main_only_mode_runs_preflight_and_atlas_without_local_or_quality(tmp_path):
    """Catches the 06:00 job also emitting the local report before Axios arrives."""
    actions = _run_mode(tmp_path, {"RUN_MAIN_ONLY": "1"})

    assert len(actions) == 2
    assert "scripts/preflight_model_check.py" in actions[0]
    assert "scripts/briefing_runner.py" in actions[1]
    assert "config.yaml" in actions[1]
    assert "config_local.yaml" not in "\n".join(actions)
    assert not any(action.startswith("quality:") for action in actions)


def test_local_only_mode_runs_local_and_then_quality_without_preflight(tmp_path):
    """Catches the 07:00 job rerunning Atlas or auditing before local completes."""
    actions = _run_mode(tmp_path, {"RUN_LOCAL_ONLY": "1"})

    assert len(actions) == 2
    assert "scripts/briefing_runner.py" in actions[0]
    assert "config_local.yaml" in actions[0]
    assert actions[1].startswith("quality:")
    assert "preflight_model_check.py" not in "\n".join(actions)
    assert "--config" not in actions[1]
