"""Shared fixtures for tests/cli/.

The CLI NATS paths fail-close when the operator creds file is absent (B2).
Unit tests that monkeypatch the NATS client should pass --allow-anonymous in
runner.invoke args to opt out of the creds check. Tests that exercise the
fail-closed path omit --allow-anonymous.
"""

from __future__ import annotations
