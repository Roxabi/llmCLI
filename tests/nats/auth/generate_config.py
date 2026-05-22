"""Generate NATS auth config for AC-7 ACL negative tests.

Run before starting the nats-auth broker container so the config file
contains ephemeral passwords rather than hardcoded values.

Usage (CI):
    python tests/nats/auth/generate_config.py --env-file /tmp/nats-creds.env
    cat /tmp/nats-creds.env >> "$GITHUB_ENV"

Usage (local dev):
    eval $(python tests/nats/auth/generate_config.py)
    docker run -d --rm --name nats-auth -p 4223:4222 \
        -v "$PWD/tests/nats/auth/nats-server.conf:/etc/nats/nats-server.conf" \
        nats:2.11-alpine
"""

from __future__ import annotations

import argparse
import os
import secrets

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "nats-server.conf")

CONFIG_TEMPLATE = """port: 4222

authorization {{
  users = [
    {{
      user: "operator"
      password: "{op_password}"
      permissions: {{
        publish: {{
          allow: ["lyra.llm.lifecycle.>", "_INBOX.>"]
        }}
        subscribe: {{
          allow: ["lyra.llm.lifecycle.>", "_INBOX.>"]
        }}
      }}
    }}
    {{
      user: "unauthorized"
      password: "{bad_password}"
      permissions: {{
        publish: {{
          deny: [">"]
        }}
        subscribe: {{
          deny: [">"]
        }}
      }}
    }}
  ]
}}
"""


def generate_config(
    op_password: str | None = None,
    bad_password: str | None = None,
    path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, str]:
    """Write a NATS server config with the given (or generated) passwords."""
    op_password = op_password or os.environ.get("NATS_TEST_OP_PASSWORD") or secrets.token_hex(16)
    bad_password = bad_password or os.environ.get("NATS_TEST_BAD_PASSWORD") or secrets.token_hex(16)
    config = CONFIG_TEMPLATE.format(op_password=op_password, bad_password=bad_password)
    with open(path, "w") as f:
        f.write(config)
    return {"op_password": op_password, "bad_password": bad_password}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NATS auth config for AC-7 tests")
    parser.add_argument("--path", default=DEFAULT_CONFIG_PATH, help="Output config path")
    parser.add_argument("--env-file", help="Append credentials to env file")
    args = parser.parse_args()
    creds = generate_config(path=args.path)
    print(f"NATS_TEST_OP_PASSWORD={creds['op_password']}")
    print(f"NATS_TEST_BAD_PASSWORD={creds['bad_password']}")
    if args.env_file:
        with open(args.env_file, "a") as f:
            f.write(f"NATS_TEST_OP_PASSWORD={creds['op_password']}\n")
            f.write(f"NATS_TEST_BAD_PASSWORD={creds['bad_password']}\n")
