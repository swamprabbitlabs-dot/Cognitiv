"""
cognitiv.brain — The CognitiveBrain orchestrator.

This is the primary public API. It owns the appraisal system, emotion
system, and memory graph, and wires them together into the cognitive loop:

    Perceive → Appraise → Emote → Remember → (Behave)

Usage:
    brain = CognitiveBrain(BrainConfig())
    brain.add_goal(Goal("wealth", "Accumulate wealth through trade", 0.8))
    brain.add_standard(Standard("honesty", "Always be truthful", 0.9))

    # Feed an event into the cognitive loop
    impulses = brain.perceive(Stimulus(
        description="Marcus accused you of cheating on the grain deal",
        category="action",
        actor_id="marcus",
        entity_ids=["marcus"],
        location_id="forum",
        timestamp=100.0,
    ))

    # Retrieve memories for an LLM prompt
    prompt_block = brain.get_memory_prompt_block("Tell me about Marcus")

    # Advance simulation time
    brain.tick(delta_time=1.0)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Callable, Optional

from .config import BrainConfig
from .emotion import AppraisalSystem, EmotionState, EmotionSystem, LLMCallback
from .memory import MemoryGraph
from .types import (
    Attitude,
    ContextSnapshot,
    EmotionalImpulse,
    Goal,
    MemoryNode,
    RetrievedMemory,
    Standard,
    Stimulus,
)


class CognitiveBrain:
    """The top-level cognitive architecture for an agent.

    Manages the interaction between appraisal, emotion, and memory
    subsystems. Provides a clean API for feeding stimuli, querying
    memories, and observing the agent's cognitive state.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        self._config = config or BrainConfig()

        # Subsystems
        self._emotion = EmotionSystem(self._config.emotion)
        self._appraisal = AppraisalSystem(self._config.appraisal)
        self._memory = MemoryGraph(self._config.memory)

        # Agent identity
        self._goals: list[Goal] = []
        self._standards: list[Standard] = []
        self._attitudes: dict[str, Attitude] = {}
        self._traits: dict[str, float] = {}

        # Context tracking
        self._current_location: str | None = None
        self._nearby_entities: list[str] = []
        self._current_activity: str | None = None
        self._world_time: float = 0.0

        # Priming timer
        self._priming_timer: float = 0.0

        # History tracking
        self._impulse_history: list[EmotionalImpulse] = []
        self._perception_count: int = 0

        # LLM callback (optional)
        self._llm_callback: LLMCallback | None = None

    # ─────────────────────────────────────────
    # Setup: Agent Identity
    # ─────────────────────────────────────────

    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the agent's motivational structure."""
        # Replace if already exists
        self._goals = [g for g in self._goals if g.id != goal.id]
        self._goals.append(goal)

    def remove_goal(self, goal_id: str) -> None:
        self._goals = [g for g in self._goals if g.id != goal_id]

    def get_goals(self) -> list[Goal]:
        return list(self._goals)

    def add_standard(self, standard: Standard) -> None:
        """Add a moral/social standard to the agent."""
        self._standards = [s for s in self._standards if s.id != standard.id]
        self._standards.append(standard)

    def remove_standard(self, standard_id: str) -> None:
        self._standards = [s for s in self._standards if s.id != standard_id]

    def get_standards(self) -> list[Standard]:
        return list(self._standards)

    def set_attitude(self, attitude: Attitude) -> None:
        """Set the agent's attitude toward an entity/concept."""
        self._attitudes[attitude.target_id] = attitude

    def get_attitude(self, target_id: str) -> Attitude | None:
        return self._attitudes.get(target_id)

    def get_attitudes(self) -> dict[str, Attitude]:
        return dict(self._attitudes)

    def set_trait(self, trait_id: str, value: float) -> None:
        """Set a personality trait (arbitrary key-value)."""
        self._traits[trait_id] = value

    def get_trait(self, trait_id: str) -> float:
        return self._traits.get(trait_id, 0.0)

    # ─────────────────────────────────────────
    # Setup: Callbacks
    # ─────────────────────────────────────────

    def set_llm_callback(self, callback: LLMCallback | None) -> None:
        """Set the LLM callback for appraisal fallback and consolidation.

        Signature: (prompt: str) -> str
        """
        self._llm_callback = callback
        self._appraisal.set_llm_callback(callback)

    # ─────────────────────────────────────────
    # Context Updates
    # ─────────────────────────────────────────

    def set_location(self, location_id: str | None) -> None:
        self._current_location = location_id

    def set_nearby_entities(self, entity_ids: list[str]) -> None:
        self._nearby_entities = list(entity_ids)

    def set_activity(self, activity: str | None) -> None:
        self._current_activity = activity

    def get_current_context(self) -> ContextSnapshot:
        """Build a ContextSnapshot from the agent's current state."""
        emotions = self._emotion.get_state()
        return ContextSnapshot(
            location_id=self._current_location,
            present_entity_ids=list(self._nearby_entities),
            activity=self._current_activity,
            emotional_state=emotions.as_vector(),
            timestamp=self._world_time,
        )

    # ─────────────────────────────────────────
    # The Cognitive Loop
    # ─────────────────────────────────────────

    def tick(self, delta_time: float) -> None:
        """Advance the cognitive system by delta_time.

        Call this every simulation tick. It:
        1. Advances the world clock
        2. Decays emotions
        3. Updates mood
        4. Runs predictive priming (on interval)

        Args:
            delta_time: Simulation time elapsed since last tick.
        """
        self._world_time += delta_time

        # Emotion decay and mood update
        self._emotion.decay(delta_time)
        self._emotion.update_mood(delta_time)

        # Predictive priming
        self._priming_timer += delta_time
        if self._priming_timer >= self._config.memory.priming_interval:
            self._memory.predictive_prime(
                location_id=self._current_location,
                nearby_entity_ids=self._nearby_entities,
                emotional_state=self._emotion.get_state().as_vector(),
            )
            self._priming_timer = 0.0

    def perceive(self, stimulus: Stimulus) -> list[EmotionalImpulse]:
        """Feed a stimulus through the full cognitive loop.

        This is the primary input method. A stimulus enters perception,
        gets appraised against goals/standards/attitudes, generates
        emotional impulses, those impulses update the emotional state,
        and the experience is stored as an episodic memory tagged with
        the resulting emotional state.

        Args:
            stimulus: The perceived event, action, or entity.

        Returns:
            The emotional impulses generated by appraisal.
        """
        # Ensure timestamp
        if stimulus.timestamp == 0.0:
            stimulus.timestamp = self._world_time

        # 1. Appraisal
        impulses = self._appraisal.evaluate(
            stimulus=stimulus,
            goals=self._goals,
            standards=self._standards,
            attitudes=self._attitudes,
            mood_bias=self._emotion.get_appraisal_bias(),
            self_id=self._config.self_id,
        )

        # 2. Emotional integration
        self._emotion.apply_impulses(impulses)

        # 3. Memory formation (tagged with post-impulse emotional state)
        emotion_state = self._emotion.get_state()
        context = self.get_current_context()

        importance = self._estimate_importance(stimulus, impulses)

        self._memory.add_episodic(
            content=stimulus.description,
            entity_ids=stimulus.entity_ids,
            context=context,
            emotional_state=emotion_state.as_vector(),
            importance=importance,
            timestamp=stimulus.timestamp,
            tags={"category": stimulus.category},
        )

        # 4. Attitude drift
        for impulse in impulses:
            if impulse.target_id and impulse.target_id in self._attitudes:
                att = self._attitudes[impulse.target_id]
                if impulse.emotion_type in ("liking", "admiration", "joy"):
                    shift = impulse.intensity * self._config.appraisal.attitude_learning_rate
                    att.valence = min(1.0, att.valence + shift)
                elif impulse.emotion_type in ("disliking", "reproach", "distress"):
                    shift = impulse.intensity * self._config.appraisal.attitude_learning_rate
                    att.valence = max(-1.0, att.valence - shift)
                att.intensity = min(1.0, att.intensity + 0.01)

        # Track history
        self._impulse_history.extend(impulses)
        if len(self._impulse_history) > 200:
            self._impulse_history = self._impulse_history[-100:]
        self._perception_count += 1

        return impulses

    # ─────────────────────────────────────────
    # Memory Retrieval
    # ─────────────────────────────────────────

    def recall(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve memories relevant to a text query.

        Uses the full cognitive retrieval pipeline:
        spreading activation + emotional state + contextual reinstatement.

        Args:
            query: Natural language query text.
            max_results: Override for max number of results.

        Returns:
            List of RetrievedMemory objects, sorted by activation.
        """
        context = self.get_current_context()
        emotions = self._emotion.get_state().as_vector()

        results = self._memory.retrieve_by_text(
            query=query,
            current_context=context,
            current_emotions=emotions,
            current_time=self._world_time,
        )

        if max_results and len(results) > max_results:
            results = results[:max_results]

        return results

    def recall_about(self, entity_id: str) -> list[RetrievedMemory]:
        """Retrieve all memories about a specific entity."""
        context = self.get_current_context()
        emotions = self._emotion.get_state().as_vector()

        return self._memory.retrieve_by_entity(
            entity_id=entity_id,
            current_context=context,
            current_emotions=emotions,
            current_time=self._world_time,
        )

    def recall_with_seeds(
        self,
        seed_nodes: dict[str, float],
    ) -> list[RetrievedMemory]:
        """Retrieve memories using pre-computed seed activations.

        Use this when you have an external scorer (TF-IDF, embeddings)
        that produces initial node activations. The graph adds
        associative spread, emotional modulation, and contextual
        reinstatement on top.

        Args:
            seed_nodes: Maps node_id → initial activation [0, 1].
        """
        context = self.get_current_context()
        emotions = self._emotion.get_state().as_vector()

        return self._memory.retrieve(
            seed_nodes=seed_nodes,
            current_context=context,
            current_emotions=emotions,
            current_time=self._world_time,
        )

    def get_memory_prompt_block(
        self,
        query: str,
        max_results: int | None = None,
    ) -> str:
        """Retrieve memories and format them for LLM prompt injection.

        Returns a structured text block with vivid memories, faded
        memories, general beliefs, and current awareness — ready to
        paste into an LLM system prompt.

        Uses a two-pass retrieval strategy:
        1. Try keyword + entity + spreading activation (fast, specific)
        2. If that returns nothing, fall back to recency-weighted
           retrieval of all episodic/semantic memories (ensures the
           NPC always has *something* to work with)

        Args:
            query: The retrieval query (typically the player's utterance).
            max_results: Max memories to include.

        Returns:
            Formatted string for prompt injection.
        """
        results = self.recall(query, max_results)

        # Fallback: if keyword retrieval found nothing, return the most
        # recent/important memories so the NPC isn't amnesiac
        if not results:
            results = self._recall_recent(max_results or 6)

        if not results:
            return "<character_memory>\nNo relevant memories.\n</character_memory>"

        vivid: list[str] = []
        faded: list[str] = []
        beliefs: list[str] = []

        for rm in results:
            if rm.node.node_type == "semantic":
                beliefs.append(f"- {rm.node.content}")
            elif rm.confidence >= 0.8 and rm.activation >= 0.3:
                vivid.append(f"- {rm.node.content}")
            else:
                prefix = ""
                if rm.is_blended:
                    prefix = "[uncertain] "
                faded.append(f"- {prefix}{rm.node.content}")

        # Current awareness from priming
        awareness_items: list[str] = []
        if self._current_location:
            awareness_items.append(f"Current location: {self._current_location}")
        for eid in self._nearby_entities[:5]:
            att = self._attitudes.get(eid)
            if att and abs(att.valence) > 0.2:
                feeling = "dislike" if att.valence < 0 else "trust"
                awareness_items.append(f"{eid} is nearby (you {feeling} them)")
            else:
                awareness_items.append(f"{eid} is nearby")

        # Build the block
        sections: list[str] = ["<character_memory>"]

        if vivid:
            sections.append("<vivid_memories>")
            sections.extend(vivid)
            sections.append("</vivid_memories>")

        if faded:
            sections.append("<faded_memories>")
            sections.extend(faded)
            sections.append("</faded_memories>")

        if beliefs:
            sections.append("<general_beliefs>")
            sections.extend(beliefs)
            sections.append("</general_beliefs>")

        if awareness_items:
            sections.append("<current_awareness>")
            sections.extend(f"- {item}" for item in awareness_items)
            sections.append("</current_awareness>")

        # Emotional state summary
        emotion = self._emotion.get_state()
        active = emotion.get_active_emotions(threshold=0.2)
        if active:
            sections.append("<emotional_state>")
            for etype, intensity in sorted(active.items(), key=lambda x: -x[1]):
                level = "strongly" if intensity > 0.6 else "somewhat" if intensity > 0.3 else "slightly"
                sections.append(f"- Feeling {level} {etype} ({intensity:.1f})")
            if emotion.mood_valence < -0.2:
                sections.append("- Overall mood: negative")
            elif emotion.mood_valence > 0.2:
                sections.append("- Overall mood: positive")
            sections.append("</emotional_state>")

        sections.append("</character_memory>")

        return "\n".join(sections)

    # ─────────────────────────────────────────
    # Consolidation
    # ─────────────────────────────────────────

    def consolidate(self) -> list[str]:
        """Run memory consolidation if an LLM callback is available.

        Returns IDs of newly created schema nodes.
        """
        if self._llm_callback is None:
            return []

        def consolidation_callback(episode_texts: list[str]) -> str:
            prompt = (
                "You are summarizing a character's experiences into a "
                "general belief or pattern. Given these specific memories:\n\n"
                + "\n".join(f"- {t}" for t in episode_texts)
                + "\n\nWhat general conclusion would this character draw? "
                "Respond in first person, one sentence only."
            )
            return self._llm_callback(prompt)

        return self._memory.consolidate(
            llm_callback=consolidation_callback,
            current_time=self._world_time,
        )

    def prune_memories(self) -> list[str]:
        """Remove decayed, consolidated episodic memories."""
        return self._memory.prune(self._world_time)

    # ─────────────────────────────────────────
    # State Observation
    # ─────────────────────────────────────────

    def get_emotional_state(self) -> EmotionState:
        return self._emotion.get_state()

    def get_mood(self) -> tuple[float, float]:
        """Return (mood_valence, mood_arousal)."""
        state = self._emotion.get_state()
        return (state.mood_valence, state.mood_arousal)

    def get_memory_stats(self) -> dict:
        return self._memory.get_stats()

    def get_recent_impulses(self, n: int = 10) -> list[EmotionalImpulse]:
        return self._impulse_history[-n:]

    def get_snapshot(self) -> dict:
        """Return a complete snapshot of the cognitive state for debugging."""
        emotion = self._emotion.get_state()
        return {
            "world_time": self._world_time,
            "perception_count": self._perception_count,
            "emotion": {
                "emotions": emotion.as_vector(),
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "dominance": emotion.dominance,
                "mood_valence": emotion.mood_valence,
                "mood_arousal": emotion.mood_arousal,
                "dominant": emotion.get_dominant_emotion(),
            },
            "goals": [
                {"id": g.id, "importance": g.importance, "progress": g.progress}
                for g in self._goals
            ],
            "attitudes": {
                tid: {"valence": a.valence, "intensity": a.intensity}
                for tid, a in self._attitudes.items()
            },
            "memory": self._memory.get_stats(),
            "location": self._current_location,
            "nearby": self._nearby_entities,
        }

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire brain state to a dict (JSON-compatible)."""
        emotion = self._emotion.get_state()
        return {
            "config": {
                "self_id": self._config.self_id,
            },
            "world_time": self._world_time,
            "goals": [
                {"id": g.id, "description": g.description,
                 "importance": g.importance, "progress": g.progress,
                 "active": g.active}
                for g in self._goals
            ],
            "standards": [
                {"id": s.id, "description": s.description,
                 "strength": s.strength}
                for s in self._standards
            ],
            "attitudes": {
                tid: {"target_id": a.target_id, "valence": a.valence,
                      "intensity": a.intensity}
                for tid, a in self._attitudes.items()
            },
            "traits": dict(self._traits),
            "emotion_state": {
                "emotions": emotion.emotions,
                "mood_valence": emotion.mood_valence,
                "mood_arousal": emotion.mood_arousal,
            },
            "context": {
                "location": self._current_location,
                "nearby_entities": self._nearby_entities,
                "activity": self._current_activity,
            },
            # Memory graph serialization would go here.
            # For a full implementation, serialize nodes and edges.
            # Omitted for brevity — the MemoryGraph would have
            # its own to_dict/from_dict methods.
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # ─────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────

    def _estimate_importance(
        self,
        stimulus: Stimulus,
        impulses: list[EmotionalImpulse],
    ) -> float:
        """Estimate the importance of a stimulus for memory formation.

        Higher emotional intensity → higher importance.
        More goals/standards affected → higher importance.
        """
        if not impulses:
            return 0.3  # baseline

        # Emotional intensity
        max_intensity = max(imp.intensity for imp in impulses)

        # Number of systems engaged
        systems = len(set(imp.emotion_type for imp in impulses))
        system_bonus = min(systems * 0.1, 0.3)

        return min(0.3 + max_intensity * 0.5 + system_bonus, 1.0)

    def _recall_recent(self, max_results: int = 6) -> list[RetrievedMemory]:
        """Fallback retrieval: return recent memories sorted by importance × recency.

        Used when keyword-based retrieval returns nothing. Ensures the NPC
        always has some memories available in the prompt block rather than
        appearing amnesiac.
        """
        from .types import RetrievedMemory

        all_nodes = self._memory.get_all_nodes("episodic") + self._memory.get_all_nodes("semantic")

        if not all_nodes:
            return []

        # Score by importance × recency
        scored = []
        for node in all_nodes:
            time_since = max(self._world_time - node.creation_time, 0.01)
            recency = 1.0 / (1.0 + math.log(time_since + 1))
            score = node.importance * 0.6 + recency * 0.4
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (node, score) in enumerate(scored[:max_results]):
            results.append(RetrievedMemory(
                node=node,
                activation=score,
                retrieval_rank=rank,
                confidence=0.7,  # moderate confidence for fallback recall
                is_blended=False,
                blend_note="",
            ))

        return results

    @property
    def memory_graph(self) -> MemoryGraph:
        """Direct access to the memory graph (for advanced use / testing)."""
        return self._memory

    @property
    def emotion_system(self) -> EmotionSystem:
        """Direct access to the emotion system (for advanced use / testing)."""
        return self._emotion

    @property
    def appraisal_system(self) -> AppraisalSystem:
        """Direct access to the appraisal system (for advanced use / testing)."""
        return self._appraisal
