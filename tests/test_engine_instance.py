"""
RED-phase tests for T1.2 — EngineInstance contract (C9).

Constraint C9: base_url belongs on EngineInstance (per-instance, port-derived),
NOT on the Engine Protocol. These tests drive the refactor in T1.7.

Expected RED failures against current scaffold:
- EngineInstance lacks 'model_name' field (scaffold uses 'model')
- EngineInstance lacks 'base_url' field / property
- Engine Protocol still exposes 'base_url' (must be removed)
"""
from __future__ import annotations

import inspect

import pytest

from llmcli.engine import Engine, EngineInstance


# ---------------------------------------------------------------------------
# 1. EngineInstance dataclass fields
# ---------------------------------------------------------------------------


class TestEngineInstanceFields:
    """EngineInstance must carry pid, port, model_name, and base_url."""

    def test_has_pid_field(self) -> None:
        fields = {f.name for f in EngineInstance.__dataclass_fields__.values()}
        assert "pid" in fields

    def test_has_port_field(self) -> None:
        fields = {f.name for f in EngineInstance.__dataclass_fields__.values()}
        assert "port" in fields

    def test_has_model_name_field(self) -> None:
        # Scaffold uses 'model' — C9 requires 'model_name'
        fields = {f.name for f in EngineInstance.__dataclass_fields__.values()}
        assert "model_name" in fields, (
            "EngineInstance must use 'model_name', not 'model' (C9 naming)"
        )

    def test_has_base_url_field_or_property(self) -> None:
        # base_url must be accessible on an instance (field or computed property)
        instance = EngineInstance(pid=1234, port=8091, model_name="test-model")
        assert hasattr(instance, "base_url"), (
            "EngineInstance must expose 'base_url' as a field or property"
        )


# ---------------------------------------------------------------------------
# 2. EngineInstance.base_url value
# ---------------------------------------------------------------------------


class TestEngineInstanceBaseUrl:
    """base_url must be derived from port (and optional host) per-instance."""

    def test_base_url_contains_port(self) -> None:
        instance = EngineInstance(pid=1234, port=8091, model_name="qwen3")
        assert "8091" in instance.base_url

    def test_base_url_ends_with_v1(self) -> None:
        instance = EngineInstance(pid=1234, port=8091, model_name="qwen3")
        assert instance.base_url.endswith("/v1")

    def test_base_url_is_http(self) -> None:
        instance = EngineInstance(pid=1234, port=8091, model_name="qwen3")
        assert instance.base_url.startswith("http://")

    def test_base_url_format(self) -> None:
        instance = EngineInstance(pid=5678, port=9000, model_name="llama")
        assert instance.base_url == "http://localhost:9000/v1"


# ---------------------------------------------------------------------------
# 3. Per-instance isolation — different ports → different base_urls
# ---------------------------------------------------------------------------


class TestEngineInstancePerInstance:
    """Proves base_url is per-instance, not shared class-level state."""

    def test_different_ports_produce_different_base_urls(self) -> None:
        a = EngineInstance(pid=100, port=8091, model_name="model-a")
        b = EngineInstance(pid=101, port=8092, model_name="model-b")
        assert a.base_url != b.base_url

    def test_each_base_url_reflects_its_own_port(self) -> None:
        a = EngineInstance(pid=100, port=8091, model_name="model-a")
        b = EngineInstance(pid=101, port=8092, model_name="model-b")
        assert "8091" in a.base_url
        assert "8092" in b.base_url
        assert "8091" not in b.base_url
        assert "8092" not in a.base_url


# ---------------------------------------------------------------------------
# 4. Engine Protocol must NOT expose base_url
# ---------------------------------------------------------------------------


class TestEngineProtocolNoBaseUrl:
    """C9: base_url must NOT be on the Engine Protocol."""

    def test_engine_protocol_has_no_base_url_attribute(self) -> None:
        # Check annotations and protocol members — base_url must not be present
        protocol_members = set(dir(Engine))
        # base_url is a Protocol @property on the scaffold — this must be removed
        assert "base_url" not in vars(Engine), (
            "Engine Protocol must not define 'base_url' — belongs on EngineInstance (C9). "
            "Remove the @property from Engine and add base_url to EngineInstance."
        )

    def test_engine_protocol_methods_are_start_stop_health_only(self) -> None:
        # Engine Protocol should only expose start, stop, health
        public_methods = {
            name
            for name, obj in vars(Engine).items()
            if not name.startswith("_")
            and (callable(obj) or isinstance(obj, (classmethod, staticmethod)))
        }
        # base_url as property shows up via __annotations__ or vars — must be absent
        protocol_annotations = getattr(Engine, "__protocol_attrs__", set())
        # For a Protocol defined with @property, it appears in __abstractmethods__ or vars
        assert "base_url" not in protocol_annotations, (
            "Engine.__protocol_attrs__ must not include 'base_url'"
        )


# ---------------------------------------------------------------------------
# 5. Engine.start signature returns EngineInstance
# ---------------------------------------------------------------------------


class TestEngineStartSignature:
    """Engine.start(spec) must declare EngineInstance as its return type."""

    def test_start_return_annotation_is_engine_instance(self) -> None:
        sig = inspect.signature(Engine.start)
        return_annotation = sig.return_annotation
        assert return_annotation is EngineInstance, (
            f"Engine.start must return EngineInstance, got {return_annotation!r}"
        )

    def test_start_accepts_spec_parameter(self) -> None:
        sig = inspect.signature(Engine.start)
        params = list(sig.parameters.keys())
        # Expect: (self, spec)
        assert "spec" in params, f"Engine.start must have 'spec' parameter, got {params}"
