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


@pytest.mark.parametrize("config_name", ["config.yaml", "config_local.yaml"])
def test_production_chains_have_no_opencode_go(config_name):
    """Catches paid OpenCode Go returning to any production routing tier."""
    config_path = Path(__file__).resolve().parents[1] / config_name
    config = yaml.safe_load(config_path.read_text())
    chains = build_model_chains(config)

    for tier in ("heavy", "medium", "light"):
        assert chains[tier]
        assert all(not rung.model.startswith("opencode-go/") for rung in chains[tier])

    for tier in ("medium", "light"):
        assert chains[tier][0].backend == "nvidia"
        assert all(rung.backend in {"nvidia", "openrouter"} for rung in chains[tier])


class TestResolveBackend:
    @pytest.mark.parametrize("model,backend", [
        ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter"),
        ("nvidia-direct/nvidia/nemotron-3-super-120b-a12b", "nvidia"),
        ("opencode/muse-spark-1.3-contributor-free", "opencode"),
        ("opencode-go/deepseek-v4-pro", "opencode"),
        ("gemini/pro", "gemini"),
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
