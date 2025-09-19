"""Dr. Reddy's Pharma Sales Swarm Integration Example.

Demonstrates how to:
 1. Load a multi-agent swarm defined in YAML (`config/dr_reddys_swarm.yaml`).
 2. Invoke the compiled swarm with a thread id (persistence via checkpointer if configured).
 3. Perform a sequence of queries that exercise handoffs between agents.

Prerequisites:
  - MongoDB with collections: dr_reddys.product_info, dr_reddys.product_inventory
  - Environment variables for LLM / embedding providers (e.g. FIREWORKS_API_KEY, VOYAGE_API_KEY)
  - (Optional) Memory MCP server at http://localhost:8080/mcp if ai-memory tool is enabled
  - Install: pip install -e .

Run:
  python examples/dr_reddys_swarm_example.py --config config/dr_reddys_swarm.yaml
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

from agent_builder.swarm_loader import load_swarm_application
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run Dr. Reddy's swarm example")
    parser.add_argument(
        "--config",
        "-c",
        default="config/dr_reddys_swarm.yaml",
        help="Path to swarm YAML config",
    )
    parser.add_argument(
        "--thread-id",
        default="demo-thread-1",
        help="Thread id for conversation state persistence",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Only load and validate swarm, skip invoking demo queries",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"Config file not found: {args.config}")

    print(f"[INFO] Loading swarm from {args.config}")
    components = load_swarm_application(args.config)
    if "swarm" not in components:
        raise SystemExit("Swarm not built (missing swarm block in config)")

    swarm = components["swarm"]
    swarm_meta = components["swarm_config"]
    print("[INFO] Swarm ready -> agents=", swarm_meta["agents"], "default=", swarm_meta["default_active_agent"])

    # Helper to invoke the swarm
    def ask(user_text: str):
        print(f"\n[USER] {user_text}")
        result = swarm.invoke(
            {"messages": [("user", user_text)]},
            config={"configurable": {"thread_id": args.thread_id}},
        )
        # Result is expected to be a dict with 'messages'; we extract last content
        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        if last is None:
            print("[ASSISTANT] <no response>")
            return
        if isinstance(last, tuple) and len(last) >= 2:
            content = last[1]
        else:
            content = getattr(last, "content", str(last))
        print(f"[ASSISTANT] {content}")

    if not args.skip_demo:
        # Example interaction sequence that should trigger at least one agent handoff
        ask("Provide the recommended dosage and key safety warnings for Allerway 5 for treating patients with renal impairments.")
        # ask("Is Allerway 5 currently in stock and which distribution centers have it?")
        # ask("Summarize both the clinical info and inventory context for a sales briefing.")
    else:
        print("[INFO] --skip-demo set; not invoking sample queries.")

    print("\n[INFO] Completed demo at", datetime.utcnow().isoformat(), "Z")


if __name__ == "__main__":
    main()
