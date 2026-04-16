"""
cognitiv.emotion — Appraisal and emotion state management.

Two subsystems in one module:

1. AppraisalSystem: Evaluates stimuli against goals, standards, and
   attitudes to produce EmotionalImpulse objects. Implements a reduced
   OCC model (Ortony, Clore & Collins, 1988).

2. EmotionState: Integrates impulses into a continuous emotional state
   with exponential decay, saturation, and slow-moving mood baseline.

The appraisal system uses keyword matching by default, with an optional
LLM callback for novel stimuli where keyword confidence is low.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import AppraisalConfig, EmotionConfig
from .types import (
    Attitude,
    EmotionalImpulse,
    Goal,
    Standard,
    Stimulus,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
    HIGH_AROUSAL_EMOTIONS,
    LOW_AROUSAL_EMOTIONS,
    HIGH_DOMINANCE_EMOTIONS,
    LOW_DOMINANCE_EMOTIONS,
)


# ─────────────────────────────────────────────
# Emotion State
# ─────────────────────────────────────────────


@dataclass
class EmotionState:
    """The agent's current emotional condition.

    Contains both categorical emotions (OCC types with intensities)
    and dimensional representations (valence, arousal, dominance)
    derived from the categorical state.

    The dimensional representation is useful for:
    - Quick similarity comparisons (cosine similarity of emotion vectors)
    - Animation/expression mapping (arousal → energy, valence → smile/frown)
    - Behavior system input (high fear + low dominance → flee)
    """

    # Categorical: emotion_type → intensity [0, 1]
    emotions: dict[str, float] = field(default_factory=dict)

    # Dimensional (derived from categorical)
    valence: float = 0.0      # [-1, 1] positive to negative
    arousal: float = 0.0      # [0, 1] activation level
    dominance: float = 0.5    # [0, 1] sense of control

    # Mood: slow-moving baseline
    mood_valence: float = 0.0
    mood_arousal: float = 0.0

    def get(self, emotion_type: str) -> float:
        """Get the current intensity of a specific emotion."""
        return self.emotions.get(emotion_type, 0.0)

    def get_dominant_emotion(self) -> tuple[str, float] | None:
        """Return the highest-intensity emotion, or None if empty."""
        if not self.emotions:
            return None
        top = max(self.emotions, key=self.emotions.get)
        return (top, self.emotions[top])

    def get_active_emotions(self, threshold: float = 0.1) -> dict[str, float]:
        """Return all emotions above a threshold."""
        return {k: v for k, v in self.emotions.items() if v >= threshold}

    def as_vector(self) -> dict[str, float]:
        """Return the full emotion vector (for similarity computation)."""
        return dict(self.emotions)

    def copy(self) -> EmotionState:
        """Deep copy for snapshotting."""
        return EmotionState(
            emotions=dict(self.emotions),
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            mood_valence=self.mood_valence,
            mood_arousal=self.mood_arousal,
        )


# ─────────────────────────────────────────────
# Emotion System (Integration + Decay)
# ─────────────────────────────────────────────


class EmotionSystem:
    """Manages the continuous emotional state of an agent.

    Responsibilities:
    - Integrate impulses into the emotional state (saturation curve)
    - Decay emotions over time (exponential, per-type half-lives)
    - Update mood (slow-moving baseline that biases appraisal)
    - Compute dimensional representation (valence, arousal, dominance)

    Usage:
        system = EmotionSystem(EmotionConfig())
        system.apply_impulses([EmotionalImpulse("joy", 0.6)])
        system.decay(delta_time=1.0)
        state = system.get_state()
    """

    def __init__(self, config: EmotionConfig) -> None:
        self._config = config
        self._state = EmotionState()

        # Pre-compute decay constants from half-lives.
        # Exponential decay: I(t) = I0 * exp(-lambda * t)
        # Half-life: lambda = ln(2) / half_life
        self._decay_constants: dict[str, float] = {}
        for etype, hl in config.half_life_overrides.items():
            self._decay_constants[etype] = math.log(2) / max(hl, 0.01)
        self._default_decay = math.log(2) / max(config.default_half_life, 0.01)

    def get_state(self) -> EmotionState:
        """Return a copy of the current emotional state."""
        return self._state.copy()

    def get_appraisal_bias(self) -> float:
        """Return the mood-based appraisal bias.

        Positive mood → positive bias (ambiguous events seem benign).
        Negative mood → negative bias (ambiguous events seem threatening).
        """
        return self._state.mood_valence * self._config.mood_bias_strength

    def apply_impulses(self, impulses: list[EmotionalImpulse]) -> None:
        """Integrate emotional impulses into the current state.

        Uses a saturation curve: new = current + impulse * (1 - current).
        This ensures emotions asymptotically approach 1.0 but never exceed it.
        Multiple small impulses of the same type compound but with
        diminishing returns.
        """
        for impulse in impulses:
            current = self._state.emotions.get(impulse.emotion_type, 0.0)
            scaled = impulse.intensity * self._config.impulse_scale
            new_value = current + scaled * (1.0 - current)
            self._state.emotions[impulse.emotion_type] = min(new_value, 1.0)

        self._update_dimensions()

    def decay(self, delta_time: float) -> None:
        """Decay all active emotions by delta_time.

        Each emotion type has its own decay rate (half-life).
        Emotions that fall below the extinction threshold are removed.
        """
        to_remove = []

        for etype, intensity in self._state.emotions.items():
            lam = self._decay_constants.get(etype, self._default_decay)
            new_intensity = intensity * math.exp(-lam * delta_time)

            if new_intensity < self._config.extinction_threshold:
                to_remove.append(etype)
            else:
                self._state.emotions[etype] = new_intensity

        for etype in to_remove:
            del self._state.emotions[etype]

        self._update_dimensions()

    def update_mood(self, delta_time: float) -> None:
        """Update mood toward the current emotional valence/arousal.

        Mood is a slow-moving average of the emotional state.
        It provides a tonic baseline that persists across individual
        emotional episodes.
        """
        rate = self._config.mood_learning_rate * delta_time
        # Exponential moving average
        self._state.mood_valence += (
            self._state.valence - self._state.mood_valence
        ) * rate
        self._state.mood_arousal += (
            self._state.arousal - self._state.mood_arousal
        ) * rate

    def _update_dimensions(self) -> None:
        """Recompute valence, arousal, and dominance from categorical emotions."""
        if not self._state.emotions:
            self._state.valence = self._state.mood_valence * 0.1
            self._state.arousal = self._state.mood_arousal * 0.1
            self._state.dominance = 0.5
            return

        # Valence: positive emotions contribute +, negative contribute -
        pos = sum(
            self._state.emotions.get(e, 0.0) for e in POSITIVE_EMOTIONS
        )
        neg = sum(
            self._state.emotions.get(e, 0.0) for e in NEGATIVE_EMOTIONS
        )
        total = pos + neg
        if total > 0:
            self._state.valence = (pos - neg) / total
        else:
            self._state.valence = 0.0

        # Arousal: high-arousal emotions contribute more
        high = sum(
            self._state.emotions.get(e, 0.0) for e in HIGH_AROUSAL_EMOTIONS
        )
        low = sum(
            self._state.emotions.get(e, 0.0) for e in LOW_AROUSAL_EMOTIONS
        )
        all_intensities = sum(self._state.emotions.values())
        if all_intensities > 0:
            self._state.arousal = (high * 1.0 + low * 0.3) / all_intensities
            self._state.arousal = min(self._state.arousal, 1.0)
        else:
            self._state.arousal = 0.0

        # Dominance: high-dominance emotions vs low-dominance
        hi_dom = sum(
            self._state.emotions.get(e, 0.0) for e in HIGH_DOMINANCE_EMOTIONS
        )
        lo_dom = sum(
            self._state.emotions.get(e, 0.0) for e in LOW_DOMINANCE_EMOTIONS
        )
        dom_total = hi_dom + lo_dom
        if dom_total > 0:
            self._state.dominance = 0.5 + 0.5 * (hi_dom - lo_dom) / dom_total
        else:
            self._state.dominance = 0.5


# ─────────────────────────────────────────────
# Appraisal System
# ─────────────────────────────────────────────


# Type alias for the LLM callback.
# Signature: (prompt: str) -> str
# Blocking call that returns the LLM's response text.
# For async environments, the caller should wrap this appropriately.
LLMCallback = Callable[[str], str]


class AppraisalSystem:
    """Evaluates stimuli against the agent's goals, standards, and attitudes.

    Implements a reduced OCC model:
    - Events → appraised against goals → joy/distress/hope/fear/relief/disappointment
    - Actions → appraised against standards → pride/shame/admiration/reproach
    - Entities → appraised against attitudes → liking/disliking

    Relevance estimation uses keyword matching (fast, deterministic).
    When keyword confidence is low and an LLM callback is configured,
    falls back to LLM-assisted appraisal.

    Usage:
        appraisal = AppraisalSystem(config, llm_callback=my_llm_fn)
        impulses = appraisal.evaluate(
            stimulus, goals, standards, attitudes,
            mood_bias=emotion_system.get_appraisal_bias(),
            self_id="self"
        )
    """

    def __init__(
        self,
        config: AppraisalConfig,
        llm_callback: LLMCallback | None = None,
    ) -> None:
        self._config = config
        self._llm_callback = llm_callback

    def set_llm_callback(self, callback: LLMCallback | None) -> None:
        self._llm_callback = callback

    def evaluate(
        self,
        stimulus: Stimulus,
        goals: list[Goal],
        standards: list[Standard],
        attitudes: dict[str, Attitude],
        mood_bias: float = 0.0,
        self_id: str = "self",
    ) -> list[EmotionalImpulse]:
        """Run the full OCC appraisal on a stimulus.

        Returns a list of emotional impulses. Multiple impulses can fire
        from a single stimulus (e.g., distress from goal threat AND
        reproach from standard violation).

        Args:
            stimulus: The perceived event/action/entity.
            goals: The agent's current goals.
            standards: The agent's moral/social standards.
            attitudes: The agent's attitudes, keyed by target_id.
            mood_bias: Current mood-based bias from the emotion system.
            self_id: The agent's identifier (for distinguishing self-actions).
        """
        impulses: list[EmotionalImpulse] = []

        # ── Goal Appraisal ──
        # Applies to all stimuli (events can affect goals)
        for goal in goals:
            if not goal.active:
                continue

            relevance = self._estimate_relevance(stimulus, goal)
            if relevance < self._config.relevance_threshold:
                continue

            congruence = self._estimate_congruence(stimulus, goal, mood_bias)

            intensity = abs(congruence) * goal.importance * relevance
            intensity = min(intensity, 1.0)

            if intensity < 0.01:
                continue

            if congruence > 0:
                etype = "joy" if stimulus.is_confirmed else "hope"
            else:
                etype = "distress" if stimulus.is_confirmed else "fear"

            impulses.append(EmotionalImpulse(
                emotion_type=etype,
                intensity=intensity,
                cause_id=stimulus.id,
                target_id=stimulus.actor_id,
                timestamp=stimulus.timestamp,
            ))

        # ── Standard Appraisal ──
        # Applies when someone performed an action
        if stimulus.category == "action" and stimulus.actor_id:
            for standard in standards:
                relevance = self._estimate_standard_relevance(stimulus, standard)
                if relevance < self._config.relevance_threshold:
                    continue

                compliance = self._estimate_compliance(stimulus, standard, mood_bias)
                intensity = abs(compliance) * standard.strength * relevance
                intensity = min(intensity, 1.0)

                if intensity < 0.01:
                    continue

                is_self = stimulus.actor_id == self_id

                if compliance > 0:
                    etype = "pride" if is_self else "admiration"
                else:
                    etype = "shame" if is_self else "reproach"

                impulses.append(EmotionalImpulse(
                    emotion_type=etype,
                    intensity=intensity,
                    cause_id=stimulus.id,
                    target_id=stimulus.actor_id if not is_self else None,
                    timestamp=stimulus.timestamp,
                ))

        # ── Attitude Appraisal ──
        # Applies to all entities mentioned in the stimulus
        for entity_id in stimulus.entity_ids:
            if entity_id not in attitudes:
                continue

            attitude = attitudes[entity_id]
            if attitude.intensity < 0.05:
                continue

            # The presence of a liked entity in a negative event intensifies
            # distress; a disliked entity in a negative event can reduce it
            # (schadenfreude). For now, we just shift the attitude.
            valence_shift = self._compute_attitude_shift(stimulus, attitude)

            if abs(valence_shift) < 0.01:
                continue

            etype = "liking" if valence_shift > 0 else "disliking"
            impulses.append(EmotionalImpulse(
                emotion_type=etype,
                intensity=min(abs(valence_shift), 1.0),
                cause_id=stimulus.id,
                target_id=entity_id,
                timestamp=stimulus.timestamp,
            ))

        return impulses

    # ── Relevance Estimation ──

    def _estimate_relevance(self, stimulus: Stimulus, goal: Goal) -> float:
        """Estimate how relevant a stimulus is to a goal.

        Uses three signals:
        1. Pre-tagged relevance (stimulus.relevant_goal_ids)
        2. Keyword overlap between stimulus description and goal keywords
        3. LLM fallback if confidence is low
        """
        # Pre-tagged: game logic already determined relevance
        if goal.id in stimulus.relevant_goal_ids:
            return 1.0

        # Keyword matching
        score = self._keyword_relevance(
            stimulus.description,
            self._config.goal_keywords.get(goal.id, []),
            goal.description,
        )

        # LLM fallback
        if (
            score < self._config.llm_fallback_confidence_threshold
            and self._config.use_llm_fallback
            and self._llm_callback is not None
        ):
            llm_score = self._llm_relevance(stimulus, goal)
            if llm_score is not None:
                score = max(score, llm_score)

        return score

    def _estimate_standard_relevance(
        self, stimulus: Stimulus, standard: Standard
    ) -> float:
        """Estimate how relevant a stimulus is to a standard."""
        if standard.id in stimulus.relevant_standard_ids:
            return 1.0

        return self._keyword_relevance(
            stimulus.description,
            self._config.standard_keywords.get(standard.id, []),
            standard.description,
        )

    # ── Congruence / Compliance Estimation ──

    def _estimate_congruence(
        self, stimulus: Stimulus, goal: Goal, mood_bias: float
    ) -> float:
        """Estimate whether a stimulus helps (+) or hinders (-) a goal.

        Returns a value in [-1, 1].
        """
        # Check pre-tagged congruence in stimulus tags
        tag_key = f"congruence_{goal.id}"
        if tag_key in stimulus.tags:
            raw = stimulus.tags[tag_key]
            return max(-1.0, min(1.0, raw + mood_bias))

        # Simple heuristic: use sentiment-like keyword matching
        raw = self._keyword_sentiment(stimulus.description, goal.description)

        # LLM fallback for ambiguous cases
        if (
            abs(raw) < self._config.llm_fallback_confidence_threshold
            and self._config.use_llm_fallback
            and self._llm_callback is not None
        ):
            llm_congruence = self._llm_congruence(stimulus, goal)
            if llm_congruence is not None:
                raw = llm_congruence

        return max(-1.0, min(1.0, raw + mood_bias))

    def _estimate_compliance(
        self, stimulus: Stimulus, standard: Standard, mood_bias: float
    ) -> float:
        """Estimate whether an action upholds (+) or violates (-) a standard."""
        tag_key = f"compliance_{standard.id}"
        if tag_key in stimulus.tags:
            raw = stimulus.tags[tag_key]
            return max(-1.0, min(1.0, raw + mood_bias * 0.5))

        raw = self._keyword_sentiment(stimulus.description, standard.description)
        return max(-1.0, min(1.0, raw + mood_bias * 0.5))

    # ── Attitude Shift ──

    def _compute_attitude_shift(
        self, stimulus: Stimulus, attitude: Attitude
    ) -> float:
        """Compute how a stimulus shifts an attitude.

        Positive events involving a liked entity reinforce liking.
        Negative events involving a liked entity reduce liking.
        And vice versa for disliked entities.
        """
        # Check for pre-tagged valence
        if "valence" in stimulus.tags:
            event_valence = stimulus.tags["valence"]
        else:
            event_valence = self._keyword_sentiment(stimulus.description, "")

        # A positive event reinforces existing positive attitudes
        # and weakens negative attitudes (and vice versa)
        shift = event_valence * self._config.attitude_learning_rate

        # Cap the shift
        return max(-self._config.max_attitude_shift,
                   min(self._config.max_attitude_shift, shift))

    # ── Keyword Matching Internals ──

    def _keyword_relevance(
        self, text: str, keywords: list[str], fallback_text: str
    ) -> float:
        """Score relevance via keyword overlap.

        If domain-specific keywords are provided, use Jaccard-like overlap.
        Otherwise, fall back to word overlap with the description.
        """
        text_lower = text.lower()
        text_words = set(text_lower.split())

        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            return min(matches / max(len(keywords), 1), 1.0)

        # Fallback: word overlap with the goal/standard description
        if fallback_text:
            desc_words = set(fallback_text.lower().split())
            # Remove common stop words
            stop = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "it", "on"}
            desc_words -= stop
            text_words -= stop
            if not desc_words:
                return 0.0
            overlap = len(text_words & desc_words)
            return min(overlap / max(len(desc_words), 1), 1.0)

        return 0.0

    def _keyword_sentiment(self, text: str, context: str) -> float:
        """Very simple sentiment signal from keyword presence.

        This is deliberately crude. It's a baseline that the LLM fallback
        improves on for ambiguous cases. For many game scenarios, stimuli
        will have pre-tagged congruence/compliance values and this won't
        be called.
        """
        text_lower = text.lower()

        positive = {
            "help", "gift", "reward", "praise", "honor", "protect",
            "support", "agree", "approve", "benefit", "succeed", "win",
            "ally", "friend", "trust", "save", "generous",
        }
        negative = {
            "attack", "steal", "insult", "betray", "threaten", "accuse",
            "cheat", "lie", "destroy", "harm", "punish", "fail", "lose",
            "enemy", "distrust", "refuse", "deny", "exploit",
        }

        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)

        if pos + neg == 0:
            return 0.0

        return (pos - neg) / (pos + neg)

    # ── LLM Fallback ──

    def _llm_relevance(self, stimulus: Stimulus, goal: Goal) -> float | None:
        """Ask the LLM how relevant a stimulus is to a goal."""
        if self._llm_callback is None:
            return None

        prompt = (
            f"Rate how relevant this event is to this character's goal.\n\n"
            f"Character's goal: \"{goal.description}\"\n"
            f"Event that occurred: \"{stimulus.description}\"\n\n"
            f"Example: If the goal is \"Protect my family\" and the event is "
            f"\"A storm destroyed nearby houses\", relevance = 0.7\n\n"
            f"Respond with ONLY a single number between 0.0 (completely "
            f"irrelevant) and 1.0 (directly relevant). No other text."
        )

        try:
            response = self._llm_callback(prompt)
            # Extract first float-like substring in case of extra text
            import re
            match = re.search(r'-?[0-9]+\.?[0-9]*', response.strip())
            if match:
                return max(0.0, min(1.0, float(match.group())))
            return None
        except (ValueError, TypeError):
            return None

    def _llm_congruence(self, stimulus: Stimulus, goal: Goal) -> float | None:
        """Ask the LLM whether a stimulus helps or hinders a goal."""
        if self._llm_callback is None:
            return None

        prompt = (
            f"Does this event HELP or HINDER this character's goal?\n\n"
            f"Character's goal: \"{goal.description}\"\n"
            f"Event that occurred: \"{stimulus.description}\"\n\n"
            f"Examples:\n"
            f"- Goal: \"Accumulate wealth\" / Event: \"Received a large payment\" → 0.8\n"
            f"- Goal: \"Maintain good reputation\" / Event: \"Was publicly accused of fraud\" → -0.9\n"
            f"- Goal: \"Protect family\" / Event: \"A friend praised my son\" → 0.3\n\n"
            f"Respond with ONLY a single number from -1.0 (strongly hinders) "
            f"to 1.0 (strongly helps). No other text."
        )

        try:
            response = self._llm_callback(prompt)
            import re
            match = re.search(r'-?[0-9]+\.?[0-9]*', response.strip())
            if match:
                return max(-1.0, min(1.0, float(match.group())))
            return None
        except (ValueError, TypeError):
            return None
