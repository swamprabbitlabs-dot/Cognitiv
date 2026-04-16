"""
cognitiv.config — All tunable parameters for the cognitive architecture.

Design philosophy: every magic number lives here. No buried constants.
Researchers can sweep these parameters systematically; game designers
can create personality profiles by overriding subsets.

Default values are calibrated for a "normal" human-like agent in a
social simulation running at ~1 tick per second of game time. Adjust
decay rates and thresholds for different time scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmotionConfig:
    """Parameters for the emotion system."""

    # ── Decay ──
    # Half-lives in simulation time units (seconds, ticks, etc.)
    # After one half-life, an emotion's intensity drops to 50%.
    default_half_life: float = 30.0

    # Per-emotion half-life overrides. Emotions not listed here
    # use default_half_life.
    half_life_overrides: dict[str, float] = field(default_factory=lambda: {
        "fear": 10.0,
        "joy": 20.0,
        "distress": 40.0,
        "hope": 25.0,
        "reproach": 60.0,
        "shame": 50.0,
        "admiration": 30.0,
        "pride": 25.0,
        "disappointment": 35.0,
        "relief": 8.0,
        "liking": 120.0,      # quasi-attitudinal, very slow decay
        "disliking": 120.0,
    })

    # Emotions below this intensity are removed entirely
    extinction_threshold: float = 0.01

    # ── Mood ──
    # How quickly mood tracks emotional state (per tick).
    # Low values = mood is sluggish and stable.
    mood_learning_rate: float = 0.005

    # How strongly mood biases appraisal.
    # Negative mood shifts ambiguous appraisals negative, etc.
    mood_bias_strength: float = 0.2

    # ── Impulse Integration ──
    # When a new impulse arrives, it's integrated with the saturation
    # curve: new = current + impulse * (1 - current).
    # This multiplier scales the impulse before integration.
    impulse_scale: float = 1.0


@dataclass
class AppraisalConfig:
    """Parameters for the appraisal system."""

    # Minimum relevance score for a goal/standard to trigger appraisal
    relevance_threshold: float = 0.15

    # Whether to fall back to LLM for appraisal when rule-based
    # estimation is low-confidence
    use_llm_fallback: bool = True

    # Confidence threshold below which LLM fallback is triggered
    llm_fallback_confidence_threshold: float = 0.3

    # How quickly attitudes drift based on new experiences [0, 1]
    attitude_learning_rate: float = 0.05

    # Maximum attitude shift from a single stimulus
    max_attitude_shift: float = 0.15

    # Keywords that signal goal-relevant stimuli.
    # Users extend this with domain-specific vocabulary.
    # Format: goal_id → list of keywords
    goal_keywords: dict[str, list[str]] = field(default_factory=dict)

    # Keywords that signal standard-relevant stimuli.
    standard_keywords: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Parameters for the memory graph and retrieval."""

    # ── Spreading Activation ──
    max_hops: int = 3
    decay_per_hop: float = 0.35           # was 0.5 — reduced to prevent saturation
    activation_threshold: float = 0.05
    max_active_nodes: int = 60
    max_retrieval_results: int = 10
    retrieval_threshold: float = 0.15     # was 0.1 — higher bar for retrieval

    # ── Base-Level Activation (ACT-R) ──
    # decay_rate in the formula: B = ln(n) - d*ln(t)
    base_decay_rate: float = 0.7          # was 0.5 — recency matters more

    # Weight of importance in base activation calculation
    importance_weight: float = 0.3

    # ── Emotional Memory Effects ──
    # How much emotional arousal at encoding boosts importance
    arousal_importance_weight: float = 0.5

    # How much emotional arousal at encoding boosts decay resistance
    arousal_decay_resistance_weight: float = 0.3

    # Minimum emotion intensity to create an emotional edge
    emotional_edge_threshold: float = 0.2

    # ── State-Dependent Retrieval ──
    # Weight of emotional state match in retrieval boost
    state_match_weight: float = 0.3

    # Weight of arousal match in retrieval boost
    arousal_match_weight: float = 0.15

    # ── Contextual Reinstatement ──
    context_location_weight: float = 0.3
    context_time_weight: float = 0.1
    context_activity_weight: float = 0.2
    context_entities_weight: float = 0.3
    context_emotion_weight: float = 0.1

    # ── Priming ──
    location_priming_strength: float = 0.15
    entity_priming_strength: float = 0.10
    emotional_priming_strength: float = 0.10
    priming_interval: float = 3.0  # simulation time units between primes

    # ── Consolidation ──
    consolidation_node_threshold: int = 150
    co_activation_threshold: int = 3
    min_cluster_size: int = 3
    schema_decay_resistance: float = 0.7
    schema_activation_bonus: float = 1.2
    post_consolidation_decay_penalty: float = 0.3

    # ── Pruning ──
    prune_activation_threshold: float = 0.05
    prune_age_threshold: float = 500.0  # simulation time units

    # ── Interference ──
    # When two memories have activations within this ratio, they interfere
    interference_similarity_threshold: float = 0.20

    # ── Fan Effect ──
    # fan_factor = 1 / (1 + ln(fan_out))
    # No config needed; the formula is fixed. Listed here for documentation.


@dataclass
class BrainConfig:
    """Top-level configuration for a CognitiveBrain instance.

    Usage:
        # Default config
        brain = CognitiveBrain(BrainConfig())

        # Stoic personality
        config = BrainConfig()
        config.emotion.mood_bias_strength = 0.05
        config.emotion.half_life_overrides["reproach"] = 200.0
        brain = CognitiveBrain(config)

        # High-anxiety personality
        config = BrainConfig()
        config.emotion.half_life_overrides["fear"] = 60.0
        config.appraisal.relevance_threshold = 0.05
        brain = CognitiveBrain(config)
    """

    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    appraisal: AppraisalConfig = field(default_factory=AppraisalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Agent's self-identifier (used to distinguish self-actions in appraisal)
    self_id: str = "self"
