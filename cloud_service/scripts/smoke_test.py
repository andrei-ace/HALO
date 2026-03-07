#!/usr/bin/env python3
"""Smoke test for the HALO cloud cognitive service.

Runs in-process against the FastAPI app (no separate uvicorn needed).
Requires GOOGLE_API_KEY to be set.

Usage:
    make smoke-cloud-service
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure cloud_service package and its tests are importable when run from repo root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_cloud_service_root = os.path.dirname(_script_dir)
sys.path.insert(0, _cloud_service_root)
sys.path.insert(0, os.path.join(_cloud_service_root, "tests"))

from conftest import idle_snapshot  # noqa: E402
from halo.contracts.serde import snapshot_to_dict  # noqa: E402


async def main() -> int:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set. Get a key at https://aistudio.google.com/apikey")
        return 1

    import httpx

    from cloud_service.app import app
    from cloud_service.deps import lifespan

    transport = httpx.ASGITransport(app=app)

    async with lifespan(app), httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        print("==> GET /health")
        resp = await client.get("/health")
        resp.raise_for_status()
        health = resp.json()
        print(f"    status={health['status']}  sessions={health['sessions']}")

        # 2. Decide with idle snapshot
        snap = idle_snapshot()
        body = {
            "snapshot": snapshot_to_dict(snap),
            "operator_cmd": "What do you see?",
        }
        print("\n==> POST /decide (idle snapshot + 'What do you see?')")
        resp = await client.post("/decide", json=body)
        resp.raise_for_status()
        decide = resp.json()
        print(f"    reasoning: {decide['reasoning'][:200]}")
        print(f"    commands:  {len(decide['commands'])} command(s)")
        for cmd in decide["commands"]:
            print(f"      - {cmd['type']}: {cmd['payload']}")

        # 3. Check session state
        print(f"\n==> GET /state/{snap.arm_id}")
        resp = await client.get(f"/state/{snap.arm_id}")
        resp.raise_for_status()
        state = resp.json()
        print(f"    readiness={state['readiness']}  cursor={state['cursor']}")

    print("\n--- Smoke test PASSED ---")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
