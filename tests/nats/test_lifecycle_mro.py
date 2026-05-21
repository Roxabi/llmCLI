"""RED tests for LifecycleMixin MRO composition — issue #34, Slice 2, T11.

These tests MUST FAIL until LifecycleMixin and LIFECYCLE_SUBJECTS are implemented
in src/llmcli/nats/_lifecycle.py (Wave 2, T14/T15).

Spec trace: SC AC-6
"""

from __future__ import annotations

import pytest

from llmcli.nats._lifecycle import LIFECYCLE_SUBJECTS, LifecycleMixin
from roxabi_nats.adapter_base import NatsAdapterBase


@pytest.mark.nats
def test_extra_subjects_super_chain():
    """_extra_subjects returns LIFECYCLE_SUBJECTS merged with super()._extra_subjects().

    Negative: deleting LifecycleMixin._extra_subjects or removing the super() call
    causes this test to fail — the "stub.subject" would vanish from the set or
    LIFECYCLE_SUBJECTS would not appear, respectively.
    """

    class StubMixin:
        def _extra_subjects(self) -> list[str]:
            return ["stub.subject"]

    class A(LifecycleMixin, StubMixin, NatsAdapterBase):
        ...

    a = A(subject="x", queue_group="y", envelope_name="z", schema_version=1)

    result = set(a._extra_subjects())

    # All lifecycle subjects must be present
    assert set(LIFECYCLE_SUBJECTS).issubset(result), (
        f"Expected all LIFECYCLE_SUBJECTS in result; missing: {set(LIFECYCLE_SUBJECTS) - result}"
    )
    # Stub subject must survive the super() chain
    assert "stub.subject" in result, (
        "StubMixin._extra_subjects() was not called — super() chain broken"
    )
    assert result == {*LIFECYCLE_SUBJECTS, "stub.subject"}


def test_heartbeat_payload_merges():
    """heartbeat_payload merges LifecycleMixin fields with super().heartbeat_payload().

    Negative: deleting LifecycleMixin.heartbeat_payload or removing the super() call
    causes the lifecycle_draining key to be absent or the base payload keys to vanish.
    """

    class A(LifecycleMixin, NatsAdapterBase):
        ...

    a = A(subject="x", queue_group="y", envelope_name="z", schema_version=1)
    # Initialise lifecycle state so _draining.is_set() is callable
    a.__init_lifecycle__()

    payload = a.heartbeat_payload()

    # LifecycleMixin must inject lifecycle_draining
    assert "lifecycle_draining" in payload, (
        "lifecycle_draining key missing from heartbeat_payload — LifecycleMixin not merging"
    )
    # Value must reflect _draining event state (not yet set → False)
    assert payload["lifecycle_draining"] is False, (
        "lifecycle_draining should be False when _draining event is not set"
    )
