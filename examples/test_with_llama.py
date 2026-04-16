#!/usr/bin/env python3
"""
Interactive test: cognitiv + llama.cpp + Gemma 3

This script connects the cognitive architecture to your local Gemma 3 model
via llama.cpp and lets you interact with an NPC in real time, observing
the appraisal, emotion, memory, and retrieval systems working together.

SETUP
─────
1. Make sure llama-server is either:
   a) Already running:  llama-server -m /path/to/gemma-3-4b-it.gguf --port 8080
   b) Or provide paths below and this script will launch it for you.

2. Run:
   python examples/test_with_llama.py

WHAT IT TESTS
─────────────
- LLM-assisted appraisal (the model evaluates goal relevance and congruence)
- LLM-driven memory consolidation (the model generates schema summaries)
- Full cognitive loop with real language understanding
- Prompt block generation with emotionally-tagged memories
- Interactive conversation showing how memory and emotion evolve
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitiv import (
    CognitiveBrain,
    BrainConfig,
    EmotionConfig,
    AppraisalConfig,
    MemoryConfig,
    Goal,
    Standard,
    Attitude,
    Stimulus,
)
from cognitiv.integrations import LlamaCppBridge, LlamaCppConfig


# ═══════════════════════════════════════════════
# CONFIGURATION — Edit these paths for your setup
# ═══════════════════════════════════════════════

# Option A: Connect to an already-running server
LLAMA_SERVER_URL = "http://localhost:8080"

# Option B: Auto-launch (uncomment and set these)
# MODEL_PATH = "/path/to/gemma-3-4b-it-Q4_K_M.gguf"
# LLAMA_SERVER_PATH = "/path/to/llama-server"  # or just "llama-server"
MODEL_PATH = None
LLAMA_SERVER_PATH = None

# ═══════════════════════════════════════════════


def header(text: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def show_emotions(brain: CognitiveBrain) -> None:
    state = brain.get_emotional_state()
    active = state.get_active_emotions(threshold=0.05)
    if active:
        for etype, intensity in sorted(active.items(), key=lambda x: -x[1]):
            bar = "█" * int(intensity * 30)
            print(f"    {etype:18s} {intensity:.3f} {bar}")
    else:
        print("    (calm)")
    print(f"    {'Valence':18s} {state.valence:+.3f}")
    print(f"    {'Mood':18s} {state.mood_valence:+.3f}")


def show_retrieval(memories, label="Memories"):
    if not memories:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    for rm in memories:
        tag = rm.node.node_type[:4].upper()
        conf_bar = "██" if rm.confidence > 0.8 else "▓▓" if rm.confidence > 0.5 else "░░"
        blend = " [uncertain]" if rm.is_blended else ""
        print(f"    {conf_bar} [{tag}] act={rm.activation:.2f} "
              f"conf={rm.confidence:.1f}{blend}  \"{rm.node.content[:80]}\"")


def main():
    # ─── Connect to llama.cpp ───

    header("CONNECTING TO LLAMA.CPP")

    if MODEL_PATH and LLAMA_SERVER_PATH:
        print(f"  Launching llama-server with model: {MODEL_PATH}")
        bridge = LlamaCppBridge.from_model(
            model_path=MODEL_PATH,
            llama_server_path=LLAMA_SERVER_PATH,
            config=LlamaCppConfig(
                n_ctx=2048,
                n_gpu_layers=99,
                temperature=0.3,
                max_tokens=150,
            ),
        )
    else:
        print(f"  Connecting to existing server at {LLAMA_SERVER_URL}")
        bridge = LlamaCppBridge(base_url=LLAMA_SERVER_URL)

    if not bridge.is_healthy():
        print("\n  ✗ Server is not responding!")
        print("  Make sure llama-server is running:")
        print("    llama-server -m /path/to/gemma.gguf --port 8080")
        sys.exit(1)

    print("  ✓ Server is healthy")

    # Quick sanity check
    print("  Testing inference...")
    start = time.monotonic()
    test_response = bridge.complete("Reply with only the word 'ready'.")
    elapsed = time.monotonic() - start
    print(f"  ✓ Response: \"{test_response}\" ({elapsed:.2f}s)")

    # ─── Create the NPC ───

    header("CREATING NPC: Lucius, Roman Grain Merchant")

    config = BrainConfig(
        self_id="lucius",
        emotion=EmotionConfig(
            mood_bias_strength=0.2,
            mood_learning_rate=0.01,
        ),
        appraisal=AppraisalConfig(
            relevance_threshold=0.1,
            use_llm_fallback=True,
            llm_fallback_confidence_threshold=0.2,
            goal_keywords={
                "wealth": ["trade", "grain", "profit", "sell", "buy", "price",
                           "money", "coin", "deal", "merchant", "market",
                           "steal", "loss", "cost"],
                "reputation": ["honor", "reputation", "respect", "trust",
                               "accuse", "slander", "praise", "name",
                               "shame", "dignity", "lie"],
                "family": ["family", "son", "daughter", "wife", "home",
                           "children", "father", "mother", "house"],
            },
            standard_keywords={
                "honesty": ["lie", "cheat", "steal", "honest", "truth",
                            "deceive", "fraud", "false", "accuse",
                            "fair", "just"],
                "piety": ["gods", "temple", "prayer", "sacrifice",
                          "omen", "augury", "sacred", "ritual"],
            },
        ),
        memory=MemoryConfig(
            max_retrieval_results=8,
            co_activation_threshold=2,
            min_cluster_size=2,
        ),
    )

    brain = CognitiveBrain(config)
    brain.set_llm_callback(bridge.complete)

    # Identity
    brain.add_goal(Goal("wealth", "Accumulate wealth through the grain trade", 0.8, 0.4))
    brain.add_goal(Goal("reputation", "Be known as an honest and respected merchant", 0.7, 0.6))
    brain.add_goal(Goal("family", "Protect and provide for my wife and children", 0.9, 0.5))

    brain.add_standard(Standard("honesty", "Always deal fairly and truthfully", 0.85))
    brain.add_standard(Standard("piety", "Respect the gods and observe proper rituals", 0.6))

    brain.set_attitude(Attitude("marcus", valence=-0.2, intensity=0.4))
    brain.set_attitude(Attitude("gaius", valence=0.5, intensity=0.5))

    brain.set_location("forum")
    brain.set_nearby_entities(["marcus", "crowd"])
    brain.set_activity("trading")

    print("  ✓ NPC configured")

    # ─── Phase 1: Scripted Events (Testing Appraisal) ───

    header("PHASE 1: SCRIPTED EVENTS — Testing LLM-Assisted Appraisal")
    print("  Feeding events through the cognitive loop.")
    print("  Watch how the LLM evaluates relevance and congruence.\n")

    events = [
        Stimulus(
            description="Marcus publicly accused Lucius of using rigged grain scales at the Forum",
            category="action",
            actor_id="marcus",
            target_id="lucius",
            entity_ids=["marcus"],
            location_id="forum",
            timestamp=10.0,
        ),
        Stimulus(
            description="Gaius stood before the crowd and vouched for Lucius's honesty, shaming Marcus",
            category="action",
            actor_id="gaius",
            target_id="lucius",
            entity_ids=["gaius", "marcus"],
            location_id="forum",
            timestamp=15.0,
        ),
        Stimulus(
            description="A Sicilian grain shipment arrived at the docks, doubling the supply and dropping prices",
            category="event",
            entity_ids=["grain_supply"],
            location_id="docks",
            timestamp=25.0,
        ),
        Stimulus(
            description="Lucius's son was caught stealing bread from a temple offering",
            category="action",
            actor_id="lucius_son",
            target_id="temple",
            entity_ids=["lucius_son", "temple"],
            location_id="temple",
            timestamp=40.0,
        ),
    ]

    for i, stim in enumerate(events):
        print(f"  Event {i+1}: \"{stim.description[:70]}...\"")

        # Tick forward
        if i > 0:
            dt = stim.timestamp - events[i-1].timestamp
            brain.tick(dt)

        t_start = time.monotonic()
        impulses = brain.perceive(stim)
        t_elapsed = time.monotonic() - t_start

        if impulses:
            for imp in impulses:
                target = f" → {imp.target_id}" if imp.target_id else ""
                print(f"    → {imp.emotion_type}: {imp.intensity:.3f}{target}")
        else:
            print(f"    → (no emotional reaction)")

        print(f"    [{t_elapsed:.2f}s, {bridge.get_stats()['request_count']} LLM calls total]")

        print(f"  Emotional state:")
        show_emotions(brain)
        print()

    # ─── Phase 2: Memory Retrieval ───

    header("PHASE 2: MEMORY RETRIEVAL — Testing Spreading Activation")

    queries = [
        ("What do I know about Marcus?", "forum", ["marcus"]),
        ("What do I know about Marcus?", "home", []),  # different context
        ("Has anything happened at the temple?", "temple", []),
        ("How is my grain trade going?", "forum", ["crowd"]),
    ]

    for query, location, nearby in queries:
        brain.set_location(location)
        brain.set_nearby_entities(nearby)
        brain.tick(1.0)

        print(f"\n  Query: \"{query}\" (at {location})")
        results = brain.recall(query)
        show_retrieval(results)

    # ─── Phase 3: Consolidation ───

    header("PHASE 3: MEMORY CONSOLIDATION — LLM Schema Generation")

    # Force co-activation for clustering
    print("  Building co-activation patterns...")
    for _ in range(5):
        brain.recall("Marcus dishonesty cheating")
        brain.recall("trading problems merchants")

    print("  Running consolidation (LLM generates schemas)...\n")
    t_start = time.monotonic()
    schema_ids = brain.consolidate()
    t_elapsed = time.monotonic() - t_start

    if schema_ids:
        for sid in schema_ids:
            node = brain.memory_graph.get_node(sid)
            if node:
                print(f"  SCHEMA: \"{node.content}\"")
        print(f"\n  [{t_elapsed:.2f}s for {len(schema_ids)} schemas]")
    else:
        print("  (No clusters met consolidation threshold)")

    # ─── Phase 4: Prompt Block ───

    header("PHASE 4: LLM PROMPT BLOCK — Ready for NPC Dialogue")

    brain.set_location("forum")
    brain.set_nearby_entities(["marcus"])
    brain.tick(1.0)

    # Test with a query that doesn't directly keyword-match any memories.
    # The fallback retrieval should still populate the prompt block.
    block = brain.get_memory_prompt_block(
        "Can I trust the merchants here?"
    )
    print(block)

    # Also test with a direct keyword match for comparison
    print("\n  --- Direct match test ---")
    block2 = brain.get_memory_prompt_block("Tell me about Marcus and grain")
    print(block2[:400] + "..." if len(block2) > 400 else block2)

    # ─── Phase 5: Interactive Conversation ───

    header("PHASE 5: INTERACTIVE MODE")
    print("  You are now talking to Lucius at the Forum.")
    print("  Type events or questions. The system will show you")
    print("  appraisal, emotions, and memory at each step.")
    print("  Type 'state' to see full cognitive snapshot.")
    print("  Type 'recall <query>' to test memory retrieval.")
    print("  Type 'consolidate' to trigger consolidation.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "state":
            snapshot = brain.get_snapshot()
            print(f"\n  Time: {snapshot['world_time']:.1f}")
            print(f"  Perceptions: {snapshot['perception_count']}")
            print(f"  Graph: {snapshot['memory']['total_nodes']} nodes, "
                  f"{snapshot['memory']['total_edges']} edges")
            print(f"  Types: {snapshot['memory']['nodes_by_type']}")
            print(f"  Dominant: {snapshot['emotion']['dominant']}")
            print(f"  Mood: {snapshot['emotion']['mood_valence']:+.3f}")
            print(f"  Attitudes: {snapshot['attitudes']}")
            print(f"\n  LLM stats: {bridge.get_stats()}")
            continue

        if user_input.lower().startswith("recall "):
            query = user_input[7:]
            results = brain.recall(query)
            show_retrieval(results, f"Recall: '{query}'")
            continue

        if user_input.lower().startswith("prompt "):
            query = user_input[7:]
            block = brain.get_memory_prompt_block(query)
            print(f"\n{block}\n")
            continue

        if user_input.lower() == "consolidate":
            schemas = brain.consolidate()
            if schemas:
                for sid in schemas:
                    node = brain.memory_graph.get_node(sid)
                    if node:
                        print(f"  SCHEMA: \"{node.content}\"")
            else:
                print("  (No clusters ready for consolidation)")
            continue

        # Treat input as a stimulus
        brain.tick(2.0)  # advance time a bit

        # Auto-extract known entity names from the input
        known_entities = set(brain.get_attitudes().keys()) | {"marcus", "gaius",
            "lucius", "lucius_son", "titus"}
        input_lower = user_input.lower()
        found_entities = [eid for eid in known_entities if eid in input_lower]

        # Detect if this is an action (someone doing something) or an event
        category = "event"
        actor = None
        if found_entities:
            # If an entity is mentioned, treat it as an action by/involving them
            category = "action"
            actor = found_entities[0]

        stimulus = Stimulus(
            description=user_input,
            category=category,
            actor_id=actor,
            entity_ids=found_entities,
            location_id=brain._current_location,
        )

        t_start = time.monotonic()
        impulses = brain.perceive(stimulus)
        t_elapsed = time.monotonic() - t_start

        if impulses:
            print(f"  Appraisal ({t_elapsed:.2f}s):")
            for imp in impulses:
                target = f" → {imp.target_id}" if imp.target_id else ""
                print(f"    {imp.emotion_type}: {imp.intensity:.3f}{target}")
        else:
            print(f"  (No emotional reaction) ({t_elapsed:.2f}s)")

        show_emotions(brain)

        # Show what the prompt block would look like
        block = brain.get_memory_prompt_block(user_input)
        print(f"\n  Prompt block preview (first 300 chars):")
        print(f"  {block[:300]}")
        print()

    # ─── Cleanup ───

    header("FINAL STATE")

    snapshot = brain.get_snapshot()
    print(f"  Total perceptions: {snapshot['perception_count']}")
    print(f"  Memory graph: {snapshot['memory']['total_nodes']} nodes, "
          f"{snapshot['memory']['total_edges']} edges")
    print(f"  Final mood: {snapshot['emotion']['mood_valence']:+.3f}")
    print(f"  LLM stats: {bridge.get_stats()}")

    bridge.shutdown()
    print("\n  Done.")


if __name__ == "__main__":
    main()
