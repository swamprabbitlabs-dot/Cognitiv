#!/usr/bin/env python3
"""
Example: A Roman grain merchant NPC experiencing a series of events.

Demonstrates:
- Goal/standard/attitude setup
- Stimulus perception and emotional appraisal
- Emotional state evolution (decay, mood drift)
- Memory formation with emotional tagging
- Spreading activation retrieval
- Contextual reinstatement (location-dependent recall)
- State-dependent memory (emotional state affects retrieval)
- Memory consolidation into schemas
- Interference and blended recall
- Predictive priming
- LLM prompt block generation

Run: python -m examples.roman_merchant
"""

import sys
import os

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
    ContextSnapshot,
)


def header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def show_emotions(brain: CognitiveBrain) -> None:
    state = brain.get_emotional_state()
    active = state.get_active_emotions(threshold=0.05)
    if active:
        for etype, intensity in sorted(active.items(), key=lambda x: -x[1]):
            bar = "█" * int(intensity * 30)
            print(f"  {etype:20s} {intensity:.3f} {bar}")
    else:
        print("  (no active emotions)")
    print(f"  {'Valence':20s} {state.valence:+.3f}")
    print(f"  {'Arousal':20s} {state.arousal:.3f}")
    print(f"  {'Mood':20s} {state.mood_valence:+.3f}")


def show_memories(memories, label="Retrieved memories"):
    print(f"\n  {label} ({len(memories)} results):")
    for rm in memories:
        conf = "██" if rm.confidence > 0.8 else "▓▓" if rm.confidence > 0.5 else "░░"
        blend = " [BLENDED]" if rm.is_blended else ""
        ntype = rm.node.node_type.upper()[:4]
        print(f"    {conf} [{ntype}] act={rm.activation:.3f} "
              f"conf={rm.confidence:.2f}{blend}")
        print(f"       \"{rm.node.content}\"")
        if rm.blend_note:
            print(f"       Note: {rm.blend_note}")


def main():
    # ─────────────────────────────────────────
    # Setup: Create a Roman grain merchant
    # ─────────────────────────────────────────

    header("SETUP: Lucius the Grain Merchant")

    config = BrainConfig(
        self_id="lucius",
        emotion=EmotionConfig(
            mood_bias_strength=0.25,
            mood_learning_rate=0.01,
        ),
        appraisal=AppraisalConfig(
            relevance_threshold=0.1,
            use_llm_fallback=False,  # no LLM in this example
            goal_keywords={
                "wealth": ["trade", "grain", "profit", "sell", "buy", "price",
                           "money", "coin", "deal", "merchant", "market"],
                "reputation": ["honor", "reputation", "respect", "trust",
                               "accuse", "slander", "praise", "name"],
                "family": ["family", "son", "daughter", "wife", "home",
                           "children", "father"],
            },
            standard_keywords={
                "honesty": ["lie", "cheat", "steal", "honest", "truth",
                            "deceive", "fraud", "false", "accuse"],
                "loyalty": ["betray", "loyal", "faithful", "ally", "friend",
                            "abandon", "support"],
            },
        ),
        memory=MemoryConfig(
            max_hops=3,
            decay_per_hop=0.5,
            max_retrieval_results=8,
            co_activation_threshold=2,  # lower for demo
            min_cluster_size=2,         # lower for demo
        ),
    )

    brain = CognitiveBrain(config)

    # Goals
    brain.add_goal(Goal("wealth", "Accumulate wealth through grain trade",
                        importance=0.8, progress=0.4))
    brain.add_goal(Goal("reputation", "Maintain an honorable reputation in Rome",
                        importance=0.7, progress=0.6))
    brain.add_goal(Goal("family", "Protect and provide for my family",
                        importance=0.9, progress=0.5))

    # Standards
    brain.add_standard(Standard("honesty", "Always be truthful in dealings",
                                strength=0.8))
    brain.add_standard(Standard("loyalty", "Stand by allies and friends",
                                strength=0.7))

    # Attitudes
    brain.set_attitude(Attitude("marcus", valence=-0.1, intensity=0.3))
    brain.set_attitude(Attitude("gaius", valence=0.4, intensity=0.5))
    brain.set_attitude(Attitude("senate", valence=0.0, intensity=0.2))

    # Initial location
    brain.set_location("forum")
    brain.set_nearby_entities(["marcus", "crowd"])
    brain.set_activity("trading")

    print("Agent: Lucius, grain merchant")
    print(f"Goals: {[g.id for g in brain.get_goals()]}")
    print(f"Standards: {[s.id for s in brain.get_standards()]}")
    print(f"Attitudes: {brain.get_attitudes()}")
    print(f"Location: forum")

    # ─────────────────────────────────────────
    # Event 1: Marcus accuses Lucius of cheating
    # ─────────────────────────────────────────

    header("EVENT 1: Marcus accuses Lucius of cheating (t=10)")

    brain.tick(10.0)

    impulses = brain.perceive(Stimulus(
        description="Marcus publicly accused me of cheating on the grain weights at the Forum",
        category="action",
        actor_id="marcus",
        target_id="lucius",
        entity_ids=["marcus", "grain_deal"],
        location_id="forum",
        timestamp=10.0,
        tags={
            "congruence_reputation": -0.8,  # strongly hurts reputation
            "congruence_wealth": -0.3,      # might hurt future trade
            "compliance_honesty": -0.7,     # marcus violated honesty standard
            "valence": -0.7,                # negative event
        },
    ))

    print("\nEmotional impulses generated:")
    for imp in impulses:
        print(f"  {imp.emotion_type}: {imp.intensity:.3f} "
              f"(cause: {imp.cause_id[:8]}, target: {imp.target_id})")

    print("\nEmotional state after event:")
    show_emotions(brain)

    print(f"\nAttitude toward Marcus: {brain.get_attitude('marcus')}")

    # ─────────────────────────────────────────
    # Event 2: A friend defends Lucius
    # ─────────────────────────────────────────

    header("EVENT 2: Gaius defends Lucius publicly (t=12)")

    brain.tick(2.0)

    impulses = brain.perceive(Stimulus(
        description="Gaius stood up and defended my honor before the crowd, calling Marcus a liar",
        category="action",
        actor_id="gaius",
        target_id="lucius",
        entity_ids=["gaius", "marcus"],
        location_id="forum",
        timestamp=12.0,
        tags={
            "congruence_reputation": 0.6,
            "compliance_loyalty": 0.8,
            "valence": 0.7,
        },
    ))

    print("\nEmotional impulses generated:")
    for imp in impulses:
        print(f"  {imp.emotion_type}: {imp.intensity:.3f} "
              f"(target: {imp.target_id})")

    print("\nEmotional state after event:")
    show_emotions(brain)

    print(f"\nAttitude toward Gaius: {brain.get_attitude('gaius')}")
    print(f"Attitude toward Marcus: {brain.get_attitude('marcus')}")

    # ─────────────────────────────────────────
    # Event 3: Another merchant cheats Lucius
    # ─────────────────────────────────────────

    header("EVENT 3: A different merchant short-changes Lucius (t=30)")

    brain.tick(18.0)

    impulses = brain.perceive(Stimulus(
        description="The merchant Titus short-changed me on a grain delivery, giving only half the agreed amount",
        category="action",
        actor_id="titus",
        target_id="lucius",
        entity_ids=["titus", "grain_deal"],
        location_id="forum",
        timestamp=30.0,
        tags={
            "congruence_wealth": -0.6,
            "compliance_honesty": -0.8,
            "valence": -0.6,
        },
    ))

    print("\nEmotional impulses generated:")
    for imp in impulses:
        print(f"  {imp.emotion_type}: {imp.intensity:.3f}")

    print("\nEmotional state:")
    show_emotions(brain)

    # ─────────────────────────────────────────
    # Event 4: Yet another trade dispute
    # ─────────────────────────────────────────

    header("EVENT 4: Overcharged by a spice merchant (t=50)")

    brain.tick(20.0)

    impulses = brain.perceive(Stimulus(
        description="A spice merchant at the Forum overcharged me by double the fair price for pepper",
        category="action",
        actor_id="spice_merchant",
        target_id="lucius",
        entity_ids=["spice_merchant"],
        location_id="forum",
        timestamp=50.0,
        tags={
            "congruence_wealth": -0.4,
            "compliance_honesty": -0.5,
            "valence": -0.5,
        },
    ))

    print("\nEmotional state:")
    show_emotions(brain)

    # ─────────────────────────────────────────
    # Memory Retrieval Test 1: Ask about Marcus
    # ─────────────────────────────────────────

    header("RETRIEVAL 1: 'Tell me about Marcus' (at the Forum)")

    brain.set_location("forum")
    brain.set_nearby_entities(["marcus"])

    # Run a priming cycle
    brain.tick(1.0)

    results = brain.recall("Tell me about Marcus")
    show_memories(results)

    # ─────────────────────────────────────────
    # Memory Retrieval Test 2: Same query, different location
    # ─────────────────────────────────────────

    header("RETRIEVAL 2: Same query, but at HOME (different context)")

    brain.set_location("home")
    brain.set_nearby_entities(["wife"])
    brain.set_activity("resting")

    brain.tick(1.0)

    results = brain.recall("Tell me about Marcus")
    show_memories(results)

    print("\n  Note: Results may differ due to contextual reinstatement.")
    print("  At the Forum, Forum-encoded memories get a retrieval boost.")

    # ─────────────────────────────────────────
    # Retrieval Test 3: Ask about trade (associative)
    # ─────────────────────────────────────────

    header("RETRIEVAL 3: 'What do I know about trade?' (associative)")

    brain.set_location("forum")
    brain.set_activity("trading")
    brain.tick(1.0)

    results = brain.recall("What do I know about trade and merchants?")
    show_memories(results)

    print("\n  Note: Spreading activation should surface trade-related")
    print("  memories even if they don't all contain the word 'trade'.")

    # ─────────────────────────────────────────
    # Consolidation
    # ─────────────────────────────────────────

    header("CONSOLIDATION: Generating schemas")

    # Provide a simple mock LLM for consolidation
    def mock_llm(prompt: str) -> str:
        """Simulate LLM schema generation."""
        lower = prompt.lower()
        if "cheat" in lower or "short-change" in lower or "overcharge" in lower:
            return "I believe merchants at the Forum are generally dishonest and will cheat me if given the chance."
        if "defend" in lower or "honor" in lower:
            return "I know I can count on Gaius to stand by me when my honor is questioned."
        return "I have had many experiences that shape my view of the world."

    brain.set_llm_callback(mock_llm)

    # Force multiple retrievals to build co-activation counts
    for _ in range(5):
        brain.recall("merchants cheating")
        brain.recall("grain deal problems")

    schema_ids = brain.consolidate()
    print(f"\nSchemas created: {len(schema_ids)}")
    for sid in schema_ids:
        node = brain.memory_graph.get_node(sid)
        if node:
            print(f"  [{sid[:8]}] {node.content}")

    # ─────────────────────────────────────────
    # Post-Consolidation Retrieval
    # ─────────────────────────────────────────

    header("RETRIEVAL 4: After consolidation")

    results = brain.recall("Can I trust the merchants here?")
    show_memories(results, "Post-consolidation retrieval")

    print("\n  Note: Schema nodes (SEMA) should appear alongside")
    print("  episodic memories, providing generalized beliefs.")

    # ─────────────────────────────────────────
    # LLM Prompt Block
    # ─────────────────────────────────────────

    header("PROMPT BLOCK: Ready for LLM injection")

    prompt_block = brain.get_memory_prompt_block(
        "A new merchant approaches and offers you a grain deal"
    )
    print(prompt_block)

    # ─────────────────────────────────────────
    # Final State
    # ─────────────────────────────────────────

    header("FINAL COGNITIVE STATE")

    snapshot = brain.get_snapshot()
    print(f"\nWorld time: {snapshot['world_time']:.1f}")
    print(f"Perceptions processed: {snapshot['perception_count']}")
    print(f"Memory graph: {snapshot['memory']['total_nodes']} nodes, "
          f"{snapshot['memory']['total_edges']} edges")
    print(f"Nodes by type: {snapshot['memory']['nodes_by_type']}")
    print(f"Dominant emotion: {snapshot['emotion']['dominant']}")
    print(f"Mood valence: {snapshot['emotion']['mood_valence']:+.3f}")

    print("\nAttitude evolution:")
    for tid, att in snapshot["attitudes"].items():
        direction = "positive" if att["valence"] > 0 else "negative" if att["valence"] < 0 else "neutral"
        print(f"  {tid}: {att['valence']:+.3f} ({direction}, "
              f"intensity={att['intensity']:.2f})")


if __name__ == "__main__":
    main()
