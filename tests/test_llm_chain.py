"""Tests for the model-chain builder.

The chain orders MODELS and infers the backend from each slug's routing
prefix. The shape it replaced ordered BACKENDS and let each pick its own
model, which made a model's rung a consequence of who hosted it: a free model
reachable only through the mostly-paid transport could never be tried before
every model on the free transport had failed, and in practice never was.
"""

from pathlib import Path

import pytest
import yaml

from scripts.llm_chain import (
    DEFAULT_CHAINS,
    TIERS,
    Rung,
    apply_pins,
    build_clients,
    build_model_chains,
    chain_timeout,
    preflight_pins,
    resolve_backend,
)


PRODUCTION_CONFIGS = ["config.yaml", "config_local.yaml", "config_finance.yaml"]

# Codex first on every tier, sized by weight class; the free HTTP rungs keep
# their previous relative order behind it.
PRODUCTION_CHAINS = {
    "heavy": [
        "codex/gpt-5.6-sol",
        "nvidia-direct/nvidia/nemotron-3-ultra-550b-a55b",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/dots-studio/dots-3-note-preview:free",
    ],
    "medium": [
        "codex/gpt-5.6-terra",
        "nvidia-direct/nvidia/nemotron-3-super-120b-a12b",
        "openrouter/dots-studio/dots-3-note-preview:free",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/nex-agi/nex-n2.5-pro:free",
    ],
    "light": [
        "codex/gpt-5.6-luna",
        "nvidia-direct/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/cohere/north-mini-code:free",
        "openrouter/inclusionai/ling-3.0-flash-vl:free",
    ],
}


def _load(config_name):
    return yaml.safe_load((Path(__file__).resolve().parents[1] / config_name).read_text())


@pytest.mark.parametrize("config_name", PRODUCTION_CONFIGS)
def test_production_chains_are_codex_first_with_free_http_fallbacks(config_name):
    """All three pipelines route identically; a blanket change must touch each."""
    chains = build_model_chains(_load(config_name))

    assert {tier: [r.model for r in rungs] for tier, rungs in chains.items()} == PRODUCTION_CHAINS
    for tier in TIERS:
        assert chains[tier][0].backend == "codex"
        assert all(r.backend in {"nvidia", "openrouter"} for r in chains[tier][1:])


@pytest.mark.parametrize("config_name", PRODUCTION_CONFIGS)
def test_production_configs_route_nothing_to_opencode(config_name):
    """Catches a free or paid opencode rung returning to any production tier.

    The transport ran `opencode run --auto` over untrusted feed text; it stays
    in the tree but must be disabled and unreachable.
    """
    config = _load(config_name)

    for models in config["llm"]["chains"].values():
        assert not any(model.startswith(("opencode/", "opencode-go/")) for model in models)
    assert config["opencode"]["enabled"] is False
    assert "opencode" not in build_clients(config)


@pytest.mark.parametrize("config_name", PRODUCTION_CONFIGS)
def test_production_configs_build_a_chain_role_codex_client(config_name):
    config = _load(config_name)
    client = build_clients(config)["codex"]

    assert type(client).__name__ == "CodexClient"
    assert client.role == "chain"
    # Its own budget, well clear of the writer's `codex.max_calls_per_run`.
    assert client.max_calls == 40 > config["codex"]["max_calls_per_run"]
    assert client.max_concurrent == 3
    # Probed like any other rung: subscription-authenticated, not per-token.
    assert not any(
        "codex/".startswith(prefix) for prefix in config["llm"].get("preflight_skip") or []
    )


@pytest.mark.parametrize("config_name", PRODUCTION_CONFIGS)
def test_production_rungs_all_finish_inside_their_window(config_name, caplog, monkeypatch):
    """CompositeClient's startup check must stay quiet for the shipped chain."""
    import logging

    from scripts.composite_client import CompositeClient

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = _load(config_name)
    with caplog.at_level(logging.WARNING, logger="scripts.composite_client"):
        client = CompositeClient(
            build_clients(config), build_model_chains(config), timeout=chain_timeout(config)
        )

    assert not [r.getMessage() for r in caplog.records if r.name == "scripts.composite_client"]
    codex = client.clients["codex"]
    worst = codex.queue_timeout + CompositeClient._worst_case_seconds(codex)
    assert worst < chain_timeout(config)


def test_default_chains_never_route_to_opencode():
    """A config with no `llm.chains` must not fall back to the free CLI models."""
    chains = build_model_chains({})

    assert [chains[tier][0].model for tier in TIERS] == [
        "codex/gpt-5.6-sol", "codex/gpt-5.6-terra", "codex/gpt-5.6-luna",
    ]
    for tier in TIERS:
        assert all(r.backend != "opencode" for r in chains[tier])


class TestCodexFallsThrough:
    """A failing Codex rung must hand the call to the next rung, as any other."""

    class _Next:
        available = True

        def __init__(self):
            self.models = []

        def invoke(self, prompt, model=None, **kwargs):
            self.models.append(model)
            return "from the fallback rung"

        def get_usage_summary(self, **kwargs):
            return ""

    def _composite(self, config=None):
        from scripts.codex_client import CodexClient
        from scripts.composite_client import CompositeClient

        values = {"enabled": True, "binary": "/opt/codex"}
        values.update(config or {})
        nxt = self._Next()
        composite = CompositeClient(
            {"codex": CodexClient(values, role="chain"), "nvidia": nxt, "openrouter": nxt},
            build_model_chains({"llm": {"chains": PRODUCTION_CHAINS}}),
            timeout=5,
        )
        return composite, nxt

    def _failed(self, returncode=1):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.stdout, result.stderr, result.returncode = "", "", returncode
        return result

    def test_nonzero_exit_falls_through_to_the_next_rung(self):
        from unittest.mock import patch

        composite, nxt = self._composite()
        with (
            patch("scripts.codex_client.shutil.which", return_value="/opt/codex"),
            patch("scripts.codex_client.subprocess.run", return_value=self._failed()),
        ):
            assert composite.invoke("rank", tier="medium") == "from the fallback rung"

        assert nxt.models == ["nvidia-direct/nvidia/nemotron-3-super-120b-a12b"]
        assert "codex/gpt-5.6-terra` did not answer" in composite._render_fallback_note()

    def test_missing_binary_falls_through_without_starting_a_process(self):
        from unittest.mock import patch

        composite, nxt = self._composite()
        with (
            patch("scripts.codex_client.shutil.which", return_value=None),
            patch("scripts.codex_client.subprocess.run") as run,
        ):
            assert composite.invoke("summarize", tier="light") == "from the fallback rung"
        run.assert_not_called()

    def test_exhausted_budget_falls_through(self):
        from unittest.mock import MagicMock, patch

        ok = MagicMock()
        ok.returncode, ok.stderr = 0, ""
        ok.stdout = (
            '{"type":"item.completed","item":{"type":"agent_message","text":"from codex"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'
        )
        composite, nxt = self._composite({"chain": {"max_calls_per_run": 1}})
        with (
            patch("scripts.codex_client.shutil.which", return_value="/opt/codex"),
            patch("scripts.codex_client.subprocess.run", return_value=ok) as run,
        ):
            assert composite.invoke("a", tier="heavy") == "from codex"
            assert composite.invoke("b", tier="heavy") == "from the fallback rung"
        run.assert_called_once()

    def test_every_rung_failing_returns_none_for_the_deterministic_path(self):
        from unittest.mock import patch

        composite, nxt = self._composite()
        nxt.invoke = lambda *args, **kwargs: None
        with (
            patch("scripts.codex_client.shutil.which", return_value="/opt/codex"),
            patch("scripts.codex_client.subprocess.run", return_value=self._failed()),
        ):
            assert composite.invoke("rank", tier="medium") is None


class TestResolveBackend:
    @pytest.mark.parametrize("model,backend", [
        ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter"),
        ("nvidia-direct/nvidia/nemotron-3-super-120b-a12b", "nvidia"),
        ("opencode/muse-spark-1.3-contributor-free", "opencode"),
        ("opencode-go/deepseek-v4-pro", "opencode"),
        ("gemini/pro", "gemini"),
        ("codex/gpt-5.6-sol", "codex"),
        ("codex/gpt-5.6-terra", "codex"),
        ("codex/gpt-5.6-luna", "codex"),
    ])
    def test_known_prefixes(self, model, backend):
        assert resolve_backend(model) == backend

    def test_longest_prefix_wins(self):
        """`opencode-go/` must not be swallowed by the `opencode/` rule."""
        assert resolve_backend("opencode-go/minimax-m3") == "opencode"
        assert resolve_backend("opencode/minimax-m3") == "opencode"

    def test_unknown_prefix_is_not_guessed(self):
        """Guessing wrong on a paid prefix spends money."""
        assert resolve_backend("mystery/model") is None
        # A bare Codex model id carries no routing prefix and is not routable.
        assert resolve_backend("gpt-5.6-sol") is None
        assert resolve_backend("nvidia/nemotron:free") is None


class TestBuildModelChains:
    def test_reads_the_configured_order_verbatim(self):
        cfg = {"llm": {"chains": {"heavy": [
            "opencode/muse-spark-1.3-contributor-free",
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        ]}}}
        assert [r.model for r in build_model_chains(cfg)["heavy"]] == [
            "opencode/muse-spark-1.3-contributor-free",
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        ]

    def test_a_tier_may_interleave_backends(self):
        cfg = {"llm": {"chains": {"heavy": [
            "opencode/muse-spark",
            "openrouter/nemotron:free",
            "opencode-go/deepseek-v4-pro",
        ]}}}
        assert [r.backend for r in build_model_chains(cfg)["heavy"]] == [
            "opencode", "openrouter", "opencode",
        ]

    def test_unroutable_rung_is_dropped_not_guessed(self):
        cfg = {"llm": {"chains": {"heavy": ["mystery/x", "openrouter/h:free"]}}}
        assert [r.model for r in build_model_chains(cfg)["heavy"]] == ["openrouter/h:free"]

    def test_duplicate_rung_is_collapsed(self):
        cfg = {"llm": {"chains": {"heavy": ["openrouter/h:free", "openrouter/h:free"]}}}
        assert len(build_model_chains(cfg)["heavy"]) == 1

    def test_missing_tier_falls_back_to_defaults(self):
        chains = build_model_chains({"llm": {"chains": {"heavy": ["openrouter/h:free"]}}})
        assert [r.model for r in chains["light"]] == DEFAULT_CHAINS["light"]

    def test_every_tier_is_present(self):
        assert set(build_model_chains({})) == set(TIERS)


class TestBuildClients:
    def test_only_enabled_backends_are_built(self):
        cfg = {"openrouter": {"enabled": True, "api_key": "k"},
               "opencode": {"enabled": False},
               "gemini": {"enabled": False}}
        assert set(build_clients(cfg)) == {"openrouter"}

    def test_builds_direct_nvidia_backend(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        cfg = {
            "openrouter": {"enabled": False},
            "opencode": {"enabled": False},
            "gemini": {"enabled": False},
            "nvidia": {"enabled": True},
        }

        clients = build_clients(cfg)

        assert set(clients) == {"nvidia"}
        assert type(clients["nvidia"]).__name__ == "NvidiaClient"

    def test_codex_is_built_as_a_chain_transport_when_enabled(self):
        cfg = {"codex": {"enabled": True, "binary": "/opt/codex", "max_calls_per_run": 5}}
        clients = build_clients(cfg)

        assert set(clients) == {"codex"}
        assert type(clients["codex"]).__name__ == "CodexClient"
        assert clients["codex"].role == "chain"
        assert clients["codex"].max_calls == 40

    def test_codex_is_not_built_unless_explicitly_enabled(self):
        assert "codex" not in build_clients({"codex": {"enabled": False}})
        assert "codex" not in build_clients({"codex": {"binary": "/opt/codex"}})
        assert "codex" not in build_clients({})

    def test_clients_are_keyed_by_backend_name(self):
        cfg = {"openrouter": {"enabled": True, "api_key": "k"},
               "opencode": {"enabled": True},
               "gemini": {"enabled": False}}
        clients = build_clients(cfg)
        assert type(clients["openrouter"]).__name__ == "OpenRouterClient"
        assert type(clients["opencode"]).__name__ == "OpencodeClient"


class TestPreflightPins:
    def test_reads_the_healthy_model_per_tier(self):
        data = {"chains": {
            "heavy": {"available": True, "model": "openrouter/h:free"},
            "light": {"available": False, "model": "openrouter/l:free"},
        }}
        assert preflight_pins(data) == {"heavy": "openrouter/h:free"}

    def test_empty_when_absent(self):
        assert preflight_pins(None) == {}
        assert preflight_pins({}) == {}


class TestApplyPins:
    def _chain(self, *models):
        return {"heavy": [Rung(model=m, backend="openrouter") for m in models]}

    def test_rotates_the_chain_to_start_at_the_pin(self):
        chains = self._chain("a", "b", "c")
        pinned = apply_pins(chains, {"heavy": "b"})
        assert [r.model for r in pinned["heavy"]] == ["b", "c", "a"]

    def test_skipped_rungs_move_to_the_back_rather_than_being_dropped(self):
        """Preflight is a 05:45 snapshot, not a verdict.

        A model that was briefly 429 at probe time is often fine by the time
        the run needs it, so it stays in the chain as a last resort.
        """
        pinned = apply_pins(self._chain("a", "b", "c"), {"heavy": "c"})
        assert set(r.model for r in pinned["heavy"]) == {"a", "b", "c"}
        assert pinned["heavy"][0].model == "c"

    def test_pin_on_the_first_rung_is_a_no_op(self):
        chains = self._chain("a", "b")
        assert apply_pins(chains, {"heavy": "a"}) == chains

    def test_pin_naming_a_model_outside_the_chain_is_ignored(self):
        """A stale pin must never inject a model the chain does not list."""
        chains = self._chain("a", "b")
        assert apply_pins(chains, {"heavy": "elsewhere"}) == chains

    def test_no_pins_leaves_the_chain_alone(self):
        chains = self._chain("a", "b")
        assert apply_pins(chains, {}) == chains


class TestChainTimeout:
    def test_prefers_rung_timeout_seconds(self):
        assert chain_timeout({"llm": {"rung_timeout_seconds": 300}}) == 300

    def test_falls_back_to_the_legacy_keys(self):
        assert chain_timeout({"llm": {"fallback_timeout_seconds": 200}}) == 200
        assert chain_timeout({"composite": {"timeout_seconds": 120}}) == 120

    def test_default_when_nothing_is_configured(self):
        assert chain_timeout({}) == 240
