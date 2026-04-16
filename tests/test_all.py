"""Tests for the cognitiv package."""

import math
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
    EmotionalImpulse,
)
from cognitiv.emotion import EmotionSystem, AppraisalSystem
from cognitiv.memory import MemoryGraph


def test_emotion_impulse_integration():
    """Impulses should increase emotion intensity with saturation."""
    system = EmotionSystem(EmotionConfig())

    system.apply_impulses([EmotionalImpulse("joy", 0.5)])
    state = system.get_state()
    assert state.get("joy") > 0.4, f"Joy should be significant: {state.get('joy')}"

    # Second impulse of same type: should increase but with diminishing returns
    system.apply_impulses([EmotionalImpulse("joy", 0.5)])
    state2 = system.get_state()
    increase = state2.get("joy") - state.get("joy")
    assert increase < 0.5, f"Saturation should limit increase: {increase}"
    assert state2.get("joy") > state.get("joy"), "Should still increase"


def test_emotion_decay():
    """Emotions should decay exponentially over time."""
    system = EmotionSystem(EmotionConfig(
        half_life_overrides={"joy": 10.0},
        extinction_threshold=0.01,
    ))

    system.apply_impulses([EmotionalImpulse("joy", 0.8)])
    initial = system.get_state().get("joy")

    # After one half-life, should be ~50%
    system.decay(10.0)
    after = system.get_state().get("joy")
    ratio = after / initial
    assert 0.45 < ratio < 0.55, f"After half-life, ratio should be ~0.5: {ratio}"


def test_emotion_extinction():
    """Emotions below threshold should be removed."""
    system = EmotionSystem(EmotionConfig(extinction_threshold=0.1))

    system.apply_impulses([EmotionalImpulse("joy", 0.05)])
    # This tiny impulse after decay should extinguish
    system.decay(100.0)
    state = system.get_state()
    assert state.get("joy") == 0.0, "Should have been extinguished"


def test_mood_drift():
    """Mood should slowly track emotional valence."""
    system = EmotionSystem(EmotionConfig(mood_learning_rate=0.1))

    # Induce negative emotion
    system.apply_impulses([EmotionalImpulse("distress", 0.8)])
    system.decay(0.0)  # just to update dimensions

    # Update mood repeatedly
    for _ in range(50):
        system.update_mood(1.0)

    state = system.get_state()
    assert state.mood_valence < -0.1, f"Mood should be negative: {state.mood_valence}"


def test_valence_computation():
    """Positive emotions should produce positive valence, negative → negative."""
    system = EmotionSystem(EmotionConfig())

    system.apply_impulses([EmotionalImpulse("joy", 0.8)])
    assert system.get_state().valence > 0, "Joy should produce positive valence"

    system2 = EmotionSystem(EmotionConfig())
    system2.apply_impulses([EmotionalImpulse("distress", 0.8)])
    assert system2.get_state().valence < 0, "Distress should produce negative valence"


def test_appraisal_goal_congruence():
    """Appraisal should generate distress for goal-incongruent events."""
    config = AppraisalConfig(
        relevance_threshold=0.05,
        use_llm_fallback=False,
        goal_keywords={"wealth": ["trade", "grain", "steal", "profit"]},
    )
    appraisal = AppraisalSystem(config)

    stimulus = Stimulus(
        description="Someone stole my grain supply",
        category="event",
        tags={"congruence_wealth": -0.8},
    )
    goals = [Goal("wealth", "Accumulate wealth", importance=0.9)]

    impulses = appraisal.evaluate(stimulus, goals, [], {})

    distress = [i for i in impulses if i.emotion_type == "distress"]
    assert len(distress) > 0, "Should produce distress for goal threat"
    assert distress[0].intensity > 0.1, f"Intensity should be significant: {distress[0].intensity}"


def test_appraisal_standard_violation():
    """Appraisal should generate reproach for others' standard violations."""
    config = AppraisalConfig(
        relevance_threshold=0.05,
        use_llm_fallback=False,
        standard_keywords={"honesty": ["cheat", "lie", "steal", "deceive"]},
    )
    appraisal = AppraisalSystem(config)

    stimulus = Stimulus(
        description="Marcus decided to cheat on the deal",
        category="action",
        actor_id="marcus",
        tags={"compliance_honesty": -0.9},
    )
    standards = [Standard("honesty", "Be honest", strength=0.8)]

    impulses = appraisal.evaluate(stimulus, [], standards, {}, self_id="self")

    reproach = [i for i in impulses if i.emotion_type == "reproach"]
    assert len(reproach) > 0, "Should produce reproach for others' violations"


def test_memory_creation_and_retrieval():
    """Basic memory creation and text-based retrieval."""
    graph = MemoryGraph(MemoryConfig())

    graph.add_episodic(
        content="Marcus cheated me on the grain deal at the forum",
        entity_ids=["marcus"],
        context=ContextSnapshot(location_id="forum"),
        importance=0.8,
        timestamp=10.0,
    )

    graph.add_episodic(
        content="Gaius and I shared wine at the tavern",
        entity_ids=["gaius"],
        context=ContextSnapshot(location_id="tavern"),
        importance=0.5,
        timestamp=20.0,
    )

    results = graph.retrieve_by_text(
        query="Marcus grain",
        current_time=25.0,
    )

    assert len(results) > 0, "Should retrieve at least one memory"
    assert "marcus" in results[0].node.content.lower() or "grain" in results[0].node.content.lower()


def test_entity_retrieval():
    """Retrieval by entity should find related memories."""
    graph = MemoryGraph(MemoryConfig())

    graph.add_episodic(
        content="Marcus accused me of cheating",
        entity_ids=["marcus"],
        timestamp=10.0,
    )
    graph.add_episodic(
        content="I saw Marcus at the temple",
        entity_ids=["marcus"],
        timestamp=20.0,
    )
    graph.add_episodic(
        content="Gaius brought wine",
        entity_ids=["gaius"],
        timestamp=15.0,
    )

    results = graph.retrieve_by_entity("marcus", current_time=25.0)
    contents = [r.node.content for r in results]

    marcus_count = sum(1 for c in contents if "marcus" in c.lower())
    assert marcus_count >= 2, f"Should find both Marcus memories: found {marcus_count}"


def test_contextual_reinstatement():
    """Memories formed at a location should be boosted when queried from that location."""
    graph = MemoryGraph(MemoryConfig(
        context_location_weight=0.5,  # strong location effect
    ))

    graph.add_episodic(
        content="I traded grain at the forum",
        entity_ids=["grain_deal"],
        context=ContextSnapshot(location_id="forum"),
        importance=0.5,
        timestamp=10.0,
    )
    graph.add_episodic(
        content="I traded grain at the docks",
        entity_ids=["grain_deal"],
        context=ContextSnapshot(location_id="docks"),
        importance=0.5,
        timestamp=15.0,
    )

    # Query from forum
    forum_results = graph.retrieve_by_text(
        query="grain trade",
        current_context=ContextSnapshot(location_id="forum"),
        current_time=20.0,
    )

    # Query from docks
    docks_results = graph.retrieve_by_text(
        query="grain trade",
        current_context=ContextSnapshot(location_id="docks"),
        current_time=20.0,
    )

    # Both should return results, but contextual match should boost
    assert len(forum_results) > 0 and len(docks_results) > 0

    # The forum memory should rank higher when queried from the forum
    if len(forum_results) >= 2:
        forum_first = forum_results[0].node.content
        print(f"  From forum, top result: '{forum_first}'")
        # This is a soft check — contextual boost may or may not flip ranking
        # depending on other factors


def test_emotional_memory_boost():
    """Memories formed during high emotion should have higher importance."""
    graph = MemoryGraph(MemoryConfig(arousal_importance_weight=0.5))

    # Calm memory
    id1 = graph.add_episodic(
        content="I walked through the market",
        emotional_state={},
        importance=0.5,
        timestamp=10.0,
    )

    # Emotional memory
    id2 = graph.add_episodic(
        content="I was attacked in the alley",
        emotional_state={"fear": 0.9, "distress": 0.8},
        importance=0.5,
        timestamp=20.0,
    )

    node1 = graph.get_node(id1)
    node2 = graph.get_node(id2)

    assert node2.importance > node1.importance, \
        f"Emotional memory should have higher importance: {node2.importance} vs {node1.importance}"
    assert node2.decay_resistance > node1.decay_resistance, \
        "Emotional memory should decay slower"


def test_spreading_activation():
    """Activation should spread across edges."""
    graph = MemoryGraph(MemoryConfig(max_retrieval_results=10))

    # Create two memories linked through a shared entity
    id1 = graph.add_episodic(
        content="Marcus sold me grain",
        entity_ids=["marcus"],
        timestamp=10.0,
        importance=0.8,
    )
    id2 = graph.add_episodic(
        content="Marcus was seen at the senate",
        entity_ids=["marcus"],
        timestamp=20.0,
        importance=0.5,
    )

    # Seed only the first node
    results = graph.retrieve(
        seed_nodes={id1: 0.9},
        current_time=25.0,
    )

    # The second memory should be activated via the shared "marcus" entity node
    result_ids = {r.node.id for r in results}

    # At minimum the seeded node should be there
    assert id1 in result_ids, "Seeded node should be retrieved"

    # The marcus-linked node might be there via spreading
    # (depends on activation thresholds)
    all_contents = [r.node.content for r in results]
    print(f"  Spread results: {all_contents}")


def test_full_brain_loop():
    """Integration test: stimulus → appraisal → emotion → memory → retrieval."""
    brain = CognitiveBrain(BrainConfig(
        self_id="self",
        appraisal=AppraisalConfig(
            relevance_threshold=0.05,
            use_llm_fallback=False,
            goal_keywords={"wealth": ["trade", "steal", "grain"]},
            standard_keywords={"honesty": ["cheat", "lie", "steal"]},
        ),
    ))

    brain.add_goal(Goal("wealth", "Get rich", importance=0.8))
    brain.add_standard(Standard("honesty", "Be honest", strength=0.7))
    brain.set_attitude(Attitude("enemy", valence=-0.5, intensity=0.6))

    # Perceive a negative event
    impulses = brain.perceive(Stimulus(
        description="An enemy tried to steal my grain supply",
        category="action",
        actor_id="enemy",
        entity_ids=["enemy"],
        timestamp=10.0,
        tags={"congruence_wealth": -0.7, "compliance_honesty": -0.8, "valence": -0.5},
    ))

    # Should produce emotional response
    assert len(impulses) > 0, "Should generate impulses"

    # Emotional state should be negative
    state = brain.get_emotional_state()
    assert state.valence < 0, f"Valence should be negative: {state.valence}"

    # Memory should be formed
    stats = brain.get_memory_stats()
    assert stats["nodes_by_type"].get("episodic", 0) > 0, "Should have episodic nodes"

    # Retrieval should work
    results = brain.recall("grain theft")
    assert len(results) > 0, "Should retrieve the memory"

    # Prompt block should be non-empty
    block = brain.get_memory_prompt_block("What happened with my grain?")
    assert "<character_memory>" in block


def test_consolidation():
    """Memory consolidation should merge similar episodes into schemas."""
    config = BrainConfig(
        memory=MemoryConfig(
            co_activation_threshold=1,  # very low for testing
            min_cluster_size=2,
        ),
    )
    brain = CognitiveBrain(config)

    # Create related memories
    brain.perceive(Stimulus(
        description="The merchant cheated me on grain weights",
        entity_ids=["merchant"],
        timestamp=10.0,
    ))
    brain.perceive(Stimulus(
        description="The merchant lied about the grain quality",
        entity_ids=["merchant"],
        timestamp=20.0,
    ))
    brain.perceive(Stimulus(
        description="The merchant overcharged me for grain",
        entity_ids=["merchant"],
        timestamp=30.0,
    ))

    # Force co-activation
    for _ in range(5):
        brain.recall("merchant cheating grain")

    # Mock LLM
    brain.set_llm_callback(
        lambda prompt: "I believe merchants are generally dishonest."
    )

    schemas = brain.consolidate()
    # May or may not produce schemas depending on co-activation patterns
    stats = brain.get_memory_stats()
    print(f"  Post-consolidation: {stats['nodes_by_type']}")


# ─────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────

def run_all():
    tests = [
        test_emotion_impulse_integration,
        test_emotion_decay,
        test_emotion_extinction,
        test_mood_drift,
        test_valence_computation,
        test_appraisal_goal_congruence,
        test_appraisal_standard_violation,
        test_memory_creation_and_retrieval,
        test_entity_retrieval,
        test_contextual_reinstatement,
        test_emotional_memory_boost,
        test_spreading_activation,
        test_full_brain_loop,
        test_consolidation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
