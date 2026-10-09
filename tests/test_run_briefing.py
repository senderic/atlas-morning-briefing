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
    # Exactly the two pipelines that have finished by now. Finance starts at
    # 07:10 and is audited by its own wrapper; naming it here would report its
    # briefing missing every morning.
    assert actions[1].count("--config") == 2
    assert "config.yaml" in actions[1] and "config_local.yaml" in actions[1]
    assert "config_finance.yaml" not in actions[1]


def _run_wrapper(tmp_path, name, args=(), env=None):
    """Run one real wrapper script against a stub interpreter and stub audit."""
    root = tmp_path / "briefing"
    root.mkdir()
    wrapper = root / name
    shutil.copy2(name, wrapper)
    python = root / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    observed = tmp_path / "wrapper-actions.txt"
    python.write_text(
        "#!/bin/bash\n"
        'printf "python:%s\\n" "$*" >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
        'exit "${ATLAS_STUB_RC:-0}"\n'
    )
    python.chmod(0o755)
    if name != "run_quality_check.sh":
        quality = root / "run_quality_check.sh"
        quality.write_text(
            "#!/bin/bash\n"
            'printf "quality:%s\\n" "$*" >> "$ATLAS_WRAPPER_OBSERVATIONS"\n'
        )
        quality.chmod(0o755)
    result = subprocess.run(
        ["bash", str(wrapper), *args],
        cwd=root,
        env={**os.environ, **(env or {}), "ATLAS_WRAPPER_OBSERVATIONS": str(observed)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    actions = observed.read_text().splitlines() if observed.exists() else []
    return result, actions


def test_dry_run_skips_the_preflight_write_and_forwards_the_flag(tmp_path):
    """A dry run changes nothing cron reads: the pre-flight probe rewrites
    .model-availability.json, so it does not run."""
    result, actions = _run_wrapper(tmp_path, "run_briefing.sh", ["--dry-run"])

    assert result.returncode == 0, result.stderr
    assert "preflight_model_check.py" not in "\n".join(actions)
    runs = [a for a in actions if a.startswith("python:")]
    assert len(runs) == 2 and all(a.endswith("--dry-run") for a in runs)
    assert actions[-1].startswith("quality:") and actions[-1].endswith("--dry-run")


def test_dry_run_logs_under_its_own_journald_tags():
    """The quality check harvests feed yields from the real tags."""
    for name, tags in (
        ("run_briefing.sh", ("atlas-briefing", "local-briefing")),
        ("run_finance_briefing.sh", ("finance-briefing",)),
    ):
        script = open(name).read()
        assert 'TAG_SUFFIX="-dryrun"' in script
        for tag in tags:
            assert f'logger -t "{tag}$TAG_SUFFIX"' in script


def test_finance_wrapper_audits_finance_after_its_own_run(tmp_path):
    result, actions = _run_wrapper(tmp_path, "run_finance_briefing.sh")

    assert result.returncode == 0, result.stderr
    assert len(actions) == 2
    assert "scripts/briefing_runner.py" in actions[0] and "config_finance.yaml" in actions[0]
    assert actions[1].startswith("quality:")
    assert actions[1].count("--config") == 1 and "config_finance.yaml" in actions[1]
    # Its digest must not replace the main + local one written minutes earlier.
    assert "--digest-suffix finance" in actions[1]


def test_finance_wrapper_reports_the_briefings_status_not_the_audits(tmp_path):
    result, actions = _run_wrapper(
        tmp_path, "run_finance_briefing.sh", env={"ATLAS_STUB_RC": "1"}
    )
    assert result.returncode == 1
    assert any(a.startswith("quality:") for a in actions)


def test_finance_wrapper_dry_run_and_skip(tmp_path):
    result, actions = _run_wrapper(tmp_path, "run_finance_briefing.sh", ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert actions[0].endswith("--dry-run") and actions[1].endswith("--dry-run")


def test_finance_wrapper_honours_skip_quality_check(tmp_path):
    result, actions = _run_wrapper(
        tmp_path, "run_finance_briefing.sh", env={"SKIP_QUALITY_CHECK": "1"}
    )
    assert result.returncode == 0, result.stderr
    assert len(actions) == 1 and actions[0].startswith("python:")


def test_quality_wrapper_audits_every_pipeline_unless_told_which(tmp_path):
    result, actions = _run_wrapper(tmp_path, "run_quality_check.sh", ["--dry-run"])
    assert result.returncode == 0, result.stderr
    assert len(actions) == 1
    for name in ("config.yaml", "config_local.yaml", "config_finance.yaml"):
        assert f"/{name}" in actions[0]
    assert actions[0].endswith("--dry-run")


def test_quality_wrapper_takes_the_callers_pipelines_as_given(tmp_path):
    result, actions = _run_wrapper(
        tmp_path, "run_quality_check.sh", ["--config", "only_this.yaml", "--deep"]
    )
    assert result.returncode == 0, result.stderr
    assert actions == ["python:" + actions[0].split("python:", 1)[1]]
    assert actions[0].count("--config") == 1 and "only_this.yaml" in actions[0]
    assert actions[0].endswith("--deep")


def test_quality_wrapper_passes_on_the_checkers_exit_code(tmp_path):
    result, _ = _run_wrapper(
        tmp_path, "run_quality_check.sh", env={"ATLAS_STUB_RC": "2"}
    )
    assert result.returncode == 2
