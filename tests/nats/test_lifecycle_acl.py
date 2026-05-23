"""AC-7 ACL negative tests for lifecycle subjects — issue #66.

These tests verify that unauthorized access to lifecycle subjects is denied
by the NATS ACL configuration. They require the nats-auth broker (CI provides
it via the nats-auth service container or local docker run).

Spec trace: AC-7 (ACL grants).  Password auth is used as a test
simplification; production uses operator nkey + ACL grants.
"""

from __future__ import annotations

import asyncio
import uuid

import nats
import pytest

LIFECYCLE_SUBJECTS = (
    "lyra.llm.lifecycle.swap",
    "lyra.llm.lifecycle.stop",
    "lyra.llm.lifecycle.status",
    "lyra.llm.lifecycle.list",
    "lyra.llm.lifecycle.reload-catalog",
)


@pytest.mark.nats
@pytest.mark.parametrize("subject", LIFECYCLE_SUBJECTS)
async def test_unauthorized_publish_rejected(
    nats_auth_broker: str,
    nats_auth_creds: dict,
    subject: str,
) -> None:
    """Publisher without pub lyra.llm.lifecycle.> is blocked.

    Arrange: operator subscriber is listening on the lifecycle subject.
    Act: unauthorized client calls nc.request() on the same subject.
    Assert: request fails because publish is rejected by ACL; the operator
    subscriber never receives the message.

    nats-py raises TimeoutError when a request publish is blocked (no reply
    arrives). The server logs a permissions violation. We capture the error
    via error_cb to prove the block is ACL-driven, not broker absence.
    """
    subject = f"{subject}.{uuid.uuid4().hex[:8]}"
    op_nc = await nats.connect(
        nats_auth_broker,
        user=nats_auth_creds["op_user"],
        password=nats_auth_creds["op_password"],
        allow_reconnect=False,
    )
    try:
        op_sub = await op_nc.subscribe(subject)
        try:
            violations: list[Exception] = []

            async def _err_cb(exc: Exception) -> None:
                violations.append(exc)

            bad_nc = await nats.connect(
                nats_auth_broker,
                user=nats_auth_creds["bad_user"],
                password=nats_auth_creds["bad_password"],
                allow_reconnect=False,
                error_cb=_err_cb,
            )
            try:
                with pytest.raises(TimeoutError):
                    await bad_nc.request(subject, b"test", timeout=0.5)
            finally:
                await bad_nc.close()
                # Yield to the event loop so the background error_cb task can
                # schedule before we inspect violations.
                await asyncio.sleep(0.1)

            # Prove the failure was an ACL violation, not a missing broker.
            assert len(violations) > 0, (
                f"Expected at least one permissions violation, got: {violations}"
            )

            # Positive control: operator did not receive the blocked message.
            with pytest.raises(TimeoutError):
                await op_sub.next_msg(timeout=1.0)
        finally:
            await op_sub.unsubscribe()
    finally:
        await op_nc.close()


@pytest.mark.nats
@pytest.mark.parametrize("subject", LIFECYCLE_SUBJECTS)
async def test_unauthorized_subscriber_no_delivery(
    nats_auth_broker: str,
    nats_auth_creds: dict,
    subject: str,
) -> None:
    """Subscriber without sub lyra.llm.lifecycle.> receives no messages.

    Arrange: operator publishes to lifecycle subject.
    Act: unauthorized client subscribes to the same subject.
    Assert: no message is delivered within 1 s (server rejects the
    subscription via ACL). Positive control: authorized subscriber receives
    the message when subscribed on the operator connection.
    """
    subject = f"{subject}.{uuid.uuid4().hex[:8]}"
    op_nc = await nats.connect(
        nats_auth_broker,
        user=nats_auth_creds["op_user"],
        password=nats_auth_creds["op_password"],
        allow_reconnect=False,
    )
    try:
        # Positive control: authorized subscriber receives the message.
        good_sub = await op_nc.subscribe(subject)
        try:
            await op_nc.publish(subject, b"authorized-msg")
            msg = await good_sub.next_msg(timeout=0.5)
            assert msg.data == b"authorized-msg"
        finally:
            await good_sub.unsubscribe()

        # Negative control: unauthorized subscriber gets no delivery.
        violations: list[Exception] = []

        async def _err_cb(exc: Exception) -> None:
            violations.append(exc)

        bad_nc = await nats.connect(
            nats_auth_broker,
            user=nats_auth_creds["bad_user"],
            password=nats_auth_creds["bad_password"],
            allow_reconnect=False,
            error_cb=_err_cb,
        )
        try:
            bad_sub = await bad_nc.subscribe(subject)
            await op_nc.publish(subject, b"unauthorized-msg")
            with pytest.raises(TimeoutError):
                await bad_sub.next_msg(timeout=1.0)

        finally:
            await bad_nc.close()
            # Yield so background error_cb can fire before we leave the scope.
            await asyncio.sleep(0.1)

        # Prove the timeout was ACL-driven.
        assert len(violations) > 0, (
            f"Expected at least one permissions violation, got: {violations}"
        )
    finally:
        await op_nc.close()
