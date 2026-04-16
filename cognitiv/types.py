"""
cognitiv.types — Core data structures for the cognitive architecture.

Design principles:
- All identifiers are plain strings. No engine-specific types.
- Everything is a dataclass for easy construction, comparison, and serialization.
- Timestamps are abstract floats (could be game time, real time, simulation ticks).
- Tags/categories use strings, not enums, so downstream users can define their own
  taxonomies without modifying library code.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


def _new_id() -> str:
    """Generate a unique node/edge identifier."""
    return uuid.uuid4().hex[:12]


# ─────────────────────────────────────────────
# Agent Identity Structures
# ─────────────────────────────────────────────


@dataclass
class Goal:
    """Something the agent is trying to achieve or maintain.

    The appraisal system evaluates stimuli against active goals to produce
    goal-based emotions (joy, distress, hope, fear, relief, disappointment).

    Attributes:
        id: Unique identifier, e.g. "accumulate_wealth".
        description: Natural language description for LLM context and
            for human-readable appraisal (the LLM fallback sends this
            string to the model).
        importance: How much the agent cares about this goal [0, 1].
            Scales the intensity of emotions produced when the goal
            is affected.
        progress: Current perceived progress toward the goal [0, 1].
            Updated by game logic or by the appraisal system when
            relevant events occur.
        active: Whether the agent is currently pursuing this goal.
            Inactive goals are skipped during appraisal.
    """

    id: str
    description: str
    importance: float = 0.5
    progress: float = 0.0
    active: bool = True


@dataclass
class Standard:
    """A moral, social, or personal norm the agent holds.

    The appraisal system evaluates actions (by self or others) against
    standards to produce standard-based emotions (pride, shame,
    admiration, reproach).

    Attributes:
        id: Unique identifier, e.g. "honor_debts".
        description: Natural language description.
        strength: How strongly the agent holds this standard [0, 1].
            Scales the intensity of pride/shame/admiration/reproach.
    """

    id: str
    description: str
    strength: float = 0.5


@dataclass
class Attitude:
    """The agent's dispositional feeling toward a specific entity or concept.

    Attitudes are long-lived and drift slowly based on experience.
    The appraisal system uses attitudes to produce liking/disliking
    emotions and to bias event appraisal for entities the agent has
    strong feelings about.

    Attributes:
        target_id: What/who this attitude is about, e.g. "marcus", "grain_tax".
        valence: Negative (dislike) to positive (like) [-1, 1].
        intensity: How strongly felt, independent of direction [0, 1].
    """

    target_id: str
    valence: float = 0.0
    intensity: float = 0.0


# ─────────────────────────────────────────────
# Stimulus — The Input to Cognition
# ─────────────────────────────────────────────


@dataclass
class Stimulus:
    """A perceived event, action, or entity that enters the cognitive loop.

    This is the primary input to the system. Game logic, conversation
    parsers, or external event systems create Stimulus objects and feed
    them to CognitiveBrain.perceive().

    The stimulus is deliberately loosely structured. Not every field is
    required for every use case. An NPC overhearing a conversation might
    only have a description and entity_ids. A combat hit might have
    actor_id, target_id, and relevant_goal_ids pre-tagged.

    Attributes:
        id: Auto-generated unique identifier.
        description: Natural language description of what happened.
            This becomes the content of the episodic memory node.
        category: What kind of stimulus this is.
            - "event": something happened (goal appraisal)
            - "action": someone did something (standard appraisal)
            - "object"/"entity": an entity is perceived (attitude appraisal)
            Multiple categories can apply; the appraisal system checks all.
        actor_id: Who performed the action (if applicable).
        target_id: Who the action was directed at (if applicable).
        entity_ids: All entities involved or mentioned.
        location_id: Where this happened.
        timestamp: When this happened (abstract float).
        tags: Arbitrary key-value metadata. Use cases include
            pre-tagging relevance to specific goals, marking intensity,
            or passing game-specific context through the system.
        is_confirmed: Whether this event has definitely occurred (True)
            or is anticipated/uncertain (False). Affects whether the
            appraisal system produces joy/distress vs. hope/fear.
        relevant_goal_ids: Optional pre-tagged goal relevance.
            If provided, the appraisal system skips relevance estimation
            for these goals and evaluates congruence directly.
        relevant_standard_ids: Optional pre-tagged standard relevance.
    """

    description: str
    category: str = "event"
    id: str = field(default_factory=_new_id)
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    entity_ids: list[str] = field(default_factory=list)
    location_id: Optional[str] = None
    timestamp: float = 0.0
    tags: dict[str, float] = field(default_factory=dict)
    is_confirmed: bool = True
    relevant_goal_ids: list[str] = field(default_factory=list)
    relevant_standard_ids: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Emotional Impulse — Output of Appraisal
# ─────────────────────────────────────────────


@dataclass
class EmotionalImpulse:
    """An instantaneous emotional reaction produced by the appraisal system.

    Impulses are spikes, not states. The EmotionSystem integrates them
    into a continuous EmotionState via the saturation curve.

    Attributes:
        emotion_type: One of the OCC taxonomy types. See EMOTION_TYPES.
        intensity: Strength of this impulse [0, 1].
        cause_id: Which stimulus triggered this impulse.
        target_id: Who/what this emotion is directed at (if applicable).
            For reproach: who did the blameworthy thing.
            For liking: who is liked.
        timestamp: When this impulse was generated.
    """

    emotion_type: str
    intensity: float
    cause_id: str = ""
    target_id: Optional[str] = None
    timestamp: float = 0.0


# The canonical OCC-derived emotion taxonomy.
# Users can extend this, but the appraisal system produces these by default.
EMOTION_TYPES = {
    # Goal-based (reactions to events)
    "joy",              # a desirable event occurred
    "distress",         # an undesirable event occurred
    "hope",             # a desirable event might occur
    "fear",             # an undesirable event might occur
    "relief",           # a feared event didn't occur
    "disappointment",   # a hoped-for event didn't occur
    # Standard-based (reactions to actions)
    "pride",            # I did something admirable
    "shame",            # I did something blameworthy
    "admiration",       # someone else did something admirable
    "reproach",         # someone else did something blameworthy
    # Attitude-based (reactions to entities)
    "liking",           # positive regard strengthened
    "disliking",        # negative regard strengthened
}

# Which emotions are positive vs. negative (for valence computation)
POSITIVE_EMOTIONS = {"joy", "hope", "relief", "pride", "admiration", "liking"}
NEGATIVE_EMOTIONS = {"distress", "fear", "disappointment", "shame", "reproach", "disliking"}

# Which emotions are high-arousal vs. low-arousal
HIGH_AROUSAL_EMOTIONS = {"fear", "joy", "reproach", "pride", "distress"}
LOW_AROUSAL_EMOTIONS = {"disappointment", "relief", "liking", "disliking"}

# Which emotions convey high dominance (sense of control)
HIGH_DOMINANCE_EMOTIONS = {"pride", "reproach", "admiration", "joy"}
LOW_DOMINANCE_EMOTIONS = {"fear", "shame", "distress", "disappointment"}


# ─────────────────────────────────────────────
# Context Snapshot — Situational State
# ─────────────────────────────────────────────


@dataclass
class ContextSnapshot:
    """A snapshot of the agent's situational context at a moment in time.

    Stored on memory edges at formation time. Used for encoding
    specificity — retrieval is boosted when the current context
    matches the context at memory formation.

    Attributes:
        location_id: Where the agent is.
        time_of_day: Abstract time-of-day marker (e.g. "morning",
            "night", or a float). String for flexibility.
        present_entity_ids: Who else is present.
        activity: What the agent is doing, e.g. "trading", "debating".
        emotional_state: Dict of emotion_type → intensity at this moment.
        world_state: Arbitrary world-state variables relevant to the agent.
        timestamp: Simulation time.
    """

    location_id: Optional[str] = None
    time_of_day: Optional[str] = None
    present_entity_ids: list[str] = field(default_factory=list)
    activity: Optional[str] = None
    emotional_state: dict[str, float] = field(default_factory=dict)
    world_state: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


# ─────────────────────────────────────────────
# Memory Graph Structures
# ─────────────────────────────────────────────


@dataclass
class MemoryNode:
    """A node in the memory graph.

    Node types follow the cognitive science taxonomy:
    - "episodic": A specific event or experience.
    - "semantic": A generalized belief or schema (consolidated from episodes).
    - "entity": A person, place, faction, or object.
    - "contextual": A situational marker (location, activity).
    - "emotional": An affective state marker.

    Attributes:
        id: Unique identifier.
        node_type: One of the types above.
        content: Natural language description of this memory.
        base_activation: Resting activation level [0, 1]. Computed from
            access frequency, recency, and importance.
        current_activation: Transient activation for the current query.
            Reset to 0 at the start of each retrieval.
        creation_time: When this node was created (simulation time).
        last_access_time: When this node was last retrieved.
        access_count: How many times this node has been retrieved.
        emotional_snapshot: The agent's emotional state when this
            memory was formed. Used for state-dependent retrieval.
        importance: Assigned importance [0, 1].
        decay_resistance: Modulates how quickly this node's base
            activation decays. Higher = slower decay.
        is_consolidated: Whether this episode has been rolled into a schema.
        consolidated_into: ID of the schema node this was consolidated into.
        tags: Arbitrary metadata.
    """

    id: str = field(default_factory=_new_id)
    node_type: str = "episodic"
    content: str = ""
    base_activation: float = 0.5
    current_activation: float = 0.0
    creation_time: float = 0.0
    last_access_time: float = 0.0
    access_count: int = 0
    emotional_snapshot: dict[str, float] = field(default_factory=dict)
    importance: float = 0.5
    decay_resistance: float = 0.0
    is_consolidated: bool = False
    consolidated_into: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryEdge:
    """A directed edge in the memory graph.

    Edge types encode the relationship between connected nodes:
    - "temporal": A happened before B.
    - "causal": A caused B.
    - "associative": A and B co-occurred or are semantically similar.
    - "hierarchical": Episode belongs to schema.
    - "entity_link": Event involves entity.
    - "emotional": Event was tagged with emotional state.
    - "contextual": Event occurred in context.
    - "contradicts": A conflicts with B.

    Attributes:
        source_id: Origin node.
        target_id: Destination node.
        edge_type: Relationship type (see above).
        weight: Activation transfer coefficient [0, 1].
        context: Situational context at the time this edge was formed.
        formation_time: When this edge was created.
        bidirectional: Whether activation spreads in both directions.
    """

    source_id: str
    target_id: str
    edge_type: str = "associative"
    weight: float = 0.5
    context: ContextSnapshot = field(default_factory=ContextSnapshot)
    formation_time: float = 0.0
    bidirectional: bool = True


# ─────────────────────────────────────────────
# Retrieval Result
# ─────────────────────────────────────────────


@dataclass
class RetrievedMemory:
    """A memory returned by the retrieval system.

    Wraps a MemoryNode with retrieval metadata so the prompt formatter
    knows how to present it (confident vs. hedged, vivid vs. faded).

    Attributes:
        node: The underlying memory node.
        activation: The activation level at retrieval time.
        retrieval_rank: Position in the retrieval ranking (0 = highest).
        confidence: How confidently this memory should be stated [0, 1].
            Derived from activation level and interference detection.
        is_blended: Whether this memory was blended with interfering
            memories during retrieval.
        blend_note: If blended, a note about what was uncertain.
    """

    node: MemoryNode
    activation: float = 0.0
    retrieval_rank: int = 0
    confidence: float = 1.0
    is_blended: bool = False
    blend_note: str = ""
