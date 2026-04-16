"""
cognitiv — A cognitive architecture for emotionally reactive agents.

Implements spreading activation memory, OCC-based emotional appraisal,
mood dynamics, memory consolidation, and interference modeling.

Quick start:
    from cognitiv import CognitiveBrain, BrainConfig, Goal, Standard, Stimulus

    brain = CognitiveBrain()
    brain.add_goal(Goal("survive", "Stay alive", importance=1.0))
    brain.perceive(Stimulus(description="A wolf appeared nearby", category="event"))
    print(brain.get_emotional_state().get_dominant_emotion())
"""

from .brain import CognitiveBrain
from .config import AppraisalConfig, BrainConfig, EmotionConfig, MemoryConfig
from .emotion import AppraisalSystem, EmotionState, EmotionSystem
from .memory import MemoryGraph
from .types import (
    Attitude,
    ContextSnapshot,
    EmotionalImpulse,
    Goal,
    MemoryEdge,
    MemoryNode,
    RetrievedMemory,
    Standard,
    Stimulus,
    EMOTION_TYPES,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
)

__version__ = "0.1.0"
__all__ = [
    "CognitiveBrain",
    "BrainConfig",
    "EmotionConfig",
    "AppraisalConfig",
    "MemoryConfig",
    "AppraisalSystem",
    "EmotionState",
    "EmotionSystem",
    "MemoryGraph",
    "Attitude",
    "ContextSnapshot",
    "EmotionalImpulse",
    "Goal",
    "MemoryEdge",
    "MemoryNode",
    "RetrievedMemory",
    "Standard",
    "Stimulus",
    "EMOTION_TYPES",
    "POSITIVE_EMOTIONS",
    "NEGATIVE_EMOTIONS",
]
