# Cognitiv

**A cognitive architecture for emotionally reactive agents.**

Cognitiv gives artificial agents human-like mental processes: emotional appraisal, associative memory, mood dynamics, and imperfect recall. It's built for LLM-driven game NPCs, behavioral research, and interactive simulations — anywhere you need agents that don't just respond but *remember, feel, and change*.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

---

## Why Cognitiv?

Most LLM agents treat memory as a database lookup: score entries by relevance, return the top matches. But human memory isn't a search index — it's associative, emotional, context-dependent, and reconstructive. A person in a marketplace already has trade memories partially activated. Their anxiety makes anxious memories more accessible. They confuse similar events. They form beliefs from patterns over time.

Cognitiv models these properties directly:

- **Spreading activation memory** — Query one concept, activation propagates across associative links to related memories
- **OCC emotional appraisal** — Events evaluated against the agent's goals, standards, and attitudes produce calibrated emotional responses
- **Contextual reinstatement** — An agent in the Forum recalls Forum memories more readily than the same agent at home
- **State-dependent recall** — Emotional state at retrieval biases which memories surface
- **Memory consolidation** — Repeated similar experiences merge into generalized schemas via LLM summarization
- **Interference and false memory** — Similar memories produce blended, naturally uncertain recall
- **Mood dynamics** — Slow-moving emotional baseline biases future appraisal, creating organic "good days" and "bad days"

The architecture is grounded in established cognitive science: ACT-R (Anderson), OCC (Ortony, Clore & Collins), encoding specificity (Tulving), reconstructive memory (Bartlett), and complementary learning systems (McClelland et al.).

---

## Installation

```bash
pip install cognitiv
```

Or install from source:

```bash
git clone https://github.com/swamprabbit-labs/cognitiv.git
cd cognitiv
pip install -e .
```

**Requirements:** Python 3.10+. Zero runtime dependencies.

---

## Quick Start

```python
from cognitiv import CognitiveBrain, Goal, Standard, Attitude, Stimulus

# Create an agent
brain = CognitiveBrain()

# Define who they are
brain.add_goal(Goal("wealth", "Accumulate wealth through trade", importance=0.8))
brain.add_standard(Standard("honesty", "Always be truthful", strength=0.9))
brain.set_attitude(Attitude("marcus", valence=-0.3, intensity=0.5))

# Feed an event through the cognitive loop
impulses = brain.perceive(Stimulus(
    description="Marcus accused you of cheating on the grain deal",
    category="action",
    actor_id="marcus",
    entity_ids=["marcus"],
    tags={"congruence_wealth": -0.5, "compliance_honesty": -0.7},
))

# See the emotional response
for imp in impulses:
    print(f"{imp.emotion_type}: {imp.intensity:.2f}")
# → distress: 0.56
# → reproach: 0.56
# → disliking: 0.04

# Retrieve memories formatted for LLM prompt injection
prompt_block = brain.get_memory_prompt_block("Tell me about Marcus")
print(prompt_block)

# Advance simulation time (call every tick)
brain.tick(delta_time=1.0)
```

---

## Architecture Overview

```
                    ┌──────────────────┐
                    │   PERCEPTION     │
                    │  (Stimulus Input) │
                    └────────┬─────────┘
                             │
                             ▼
              ┌─────▶┌──────────────────┐
              │      │    APPRAISAL     │
              │      │ (OCC Evaluation)  │
              │      └────────┬─────────┘
              │               │
              │               ▼
              │      ┌──────────────────┐
              │      │     EMOTION      │
              │      │ (State + Decay)   │
              │      └──┬────┬──────────┘
              │         │    │
              │         │    ▼
              │         │   ┌──────────────────┐
              │         │   │     MEMORY       │
              │         │   │ (Graph + Recall)  │
              │         │   └────────┬─────────┘
              │         │            │
              │         ▼            ▼
              │      ┌──────────────────┐
              └──────┤    BEHAVIOR      │
                     │  (Output to LLM)  │
                     └──────────────────┘
```

Four subsystems communicate through a shared cognitive state. Every subsystem reads from and writes to the same blackboard, creating feedback loops: emotions bias future appraisal, memories are tagged with emotions at formation, and emotional state at retrieval influences which memories surface.

---

## Core Concepts

### Stimuli

A **Stimulus** is any perceivable event, action, or entity. You feed stimuli into the brain via `brain.perceive()`:

```python
from cognitiv import Stimulus

stimulus = Stimulus(
    description="A messenger brought word of victory at the frontier",
    category="event",             # "event", "action", or "object"
    actor_id="messenger",          # who did it (for actions)
    entity_ids=["messenger"],      # all entities involved
    location_id="forum",           # where it happened
    tags={
        "congruence_wealth": 0.3,  # pre-tagged appraisal hints (optional)
        "valence": 0.7,
    },
)
```

### Goals, Standards, and Attitudes

These define the agent's identity and are what the appraisal system evaluates stimuli against:

```python
from cognitiv import Goal, Standard, Attitude

# Goals: what the agent wants
#   (events produce joy/distress/hope/fear)
brain.add_goal(Goal("wealth", "Grow my trading business", importance=0.8))

# Standards: what the agent believes is right
#   (actions produce pride/shame/admiration/reproach)
brain.add_standard(Standard("honor", "Keep your word", strength=0.9))

# Attitudes: how the agent feels about specific entities
#   (produces liking/disliking)
brain.set_attitude(Attitude("marcus", valence=-0.3, intensity=0.5))
```

### Emotion Taxonomy

Cognitiv uses a 12-emotion reduced OCC taxonomy:

| Category | Positive | Negative |
|----------|----------|----------|
| **Goal-based** (events) | joy, hope, relief | distress, fear, disappointment |
| **Standard-based** (actions) | pride, admiration | shame, reproach |
| **Attitude-based** (entities) | liking | disliking |

Compound emotions emerge naturally from co-occurring impulses — anger is reproach + distress, gratitude is admiration + joy.

### Memory Retrieval

Three ways to query memory:

```python
# 1. By text (keyword + entity matching, then spreading activation)
memories = brain.recall("What do I know about Marcus?")

# 2. By specific entity
memories = brain.recall_about("marcus")

# 3. With pre-computed seeds (e.g., from TF-IDF or embeddings)
memories = brain.recall_with_seeds({"node_abc": 0.9, "node_def": 0.6})

# For LLM prompts, use the formatted block
prompt_block = brain.get_memory_prompt_block("player's utterance here")
```

The prompt block separates memories by confidence level (vivid / faded / beliefs) and includes current emotional state:

```xml
<character_memory>
<vivid_memories>
- Marcus accused me of cheating at the Forum
- Gaius defended my honor before the crowd
</vivid_memories>
<faded_memories>
- [uncertain] The merchant short-changed me on a grain delivery
</faded_memories>
<general_beliefs>
- I believe merchants at the Forum are generally dishonest.
</general_beliefs>
<current_awareness>
- Current location: forum
- marcus is nearby (you dislike them)
</current_awareness>
<emotional_state>
- Feeling strongly distress (0.8)
- Feeling somewhat reproach (0.4)
- Overall mood: negative
</emotional_state>
</character_memory>
```

---

## LLM Integration

Cognitiv uses LLMs for two things: **appraisal fallback** (evaluating relevance/congruence for novel stimuli) and **memory consolidation** (generating schema summaries from clustered episodes).

Any LLM works — you just provide a callback function:

```python
def my_llm_callback(prompt: str) -> str:
    # Call OpenAI, Anthropic, local llama.cpp, whatever
    return llm_response

brain.set_llm_callback(my_llm_callback)
```

### Included: llama.cpp Bridge

Cognitiv ships with a bridge for local GGUF models via llama.cpp:

```python
from cognitiv.integrations import LlamaCppBridge

# Connect to an already-running server
bridge = LlamaCppBridge(base_url="http://localhost:8080")

# Or auto-launch
bridge = LlamaCppBridge.from_model(
    model_path="/path/to/gemma-3-4b-it.gguf",
    llama_server_path="llama-server",
)

brain.set_llm_callback(bridge.complete)
```

Start a llama.cpp server first:

```bash
llama-server -m /path/to/model.gguf --port 8080 --ctx-size 2048 --n-gpu-layers 99
```

**Model recommendation:** Gemma 3 4B IT works well for appraisal — it reliably returns bare numeric responses. Reasoning models (like Gemma 4 E2B) tend to wrap numbers in explanatory text and degrade performance.

---

## Configuration

Every tunable parameter lives in `BrainConfig`. Defaults are calibrated for human-like agents at ~1 simulation tick per second:

```python
from cognitiv import BrainConfig, EmotionConfig, MemoryConfig

# Custom personality: stoic
stoic = BrainConfig(
    emotion=EmotionConfig(
        mood_bias_strength=0.05,           # barely affected by mood
        half_life_overrides={
            "fear": 5.0,                    # recovers from fear quickly
            "reproach": 200.0,              # holds grudges forever
            "joy": 10.0,                    # reserved positive affect
        },
    ),
)

# Custom personality: passionate
passionate = BrainConfig(
    emotion=EmotionConfig(
        mood_bias_strength=0.4,            # mood swings heavily
        half_life_overrides={
            "fear": 30.0,
            "joy": 30.0,
            "distress": 40.0,
        },
    ),
)

brain = CognitiveBrain(stoic)
```

Full config reference in [`cognitiv/config.py`](cognitiv/config.py).

---

## Examples

### Run the Roman merchant scenario

A fully scripted example showing every subsystem in action:

```bash
python -m examples.roman_merchant
```

Output traces emotional appraisal, memory formation, retrieval, consolidation, and interference detection across a multi-event scenario.

### Interactive test with a local LLM

Connect to llama.cpp and have a real conversation:

```bash
# Terminal 1: start the LLM server
llama-server -m gemma-3-4b-it.gguf --port 8080

# Terminal 2: run the interactive test
python examples/test_with_llama.py
```

This drops you into a REPL where you can type events and watch the cognitive state evolve in real time. Commands: `state` for a full snapshot, `recall <query>` to test retrieval, `prompt <query>` to see the LLM prompt block, `consolidate` to trigger schema generation.

---

## Testing

```bash
python -m tests.test_all
```

14 unit tests covering emotion dynamics, appraisal, memory graph operations, spreading activation, contextual reinstatement, consolidation, and the full cognitive loop.

---

## Project Structure

```
cognitiv/
├── cognitiv/
│   ├── __init__.py           # Public API
│   ├── types.py              # Core data structures
│   ├── config.py             # All tunable parameters
│   ├── emotion.py            # Appraisal system + emotion integration
│   ├── memory.py             # Graph, spreading activation, consolidation
│   ├── brain.py              # CognitiveBrain orchestrator
│   └── integrations/
│       └── __init__.py       # LlamaCppBridge
├── examples/
│   ├── roman_merchant.py     # Full scripted scenario
│   └── test_with_llama.py    # Interactive LLM testing
├── tests/
│   └── test_all.py           # Unit tests
├── pyproject.toml
└── README.md
```

---

## Use Cases

**Game NPCs.** Cognitiv was designed alongside [Personica AI](https://fab.com), a commercial UE5 plugin for LLM-driven NPCs. The architecture gives game characters persistent emotional states, associative memory, and naturally imperfect recall — turning static dialogue trees into agents that change based on what they've experienced.

**Behavioral research.** The pure-Python implementation with zero dependencies makes Cognitiv easy to integrate into research pipelines. Run thousands of simulated agents, sweep appraisal parameters systematically, or use it as a baseline for comparing cognitive architectures.

**Interactive fiction and chatbots.** Any agent that needs to feel like a coherent individual rather than a stateless responder benefits from emotional appraisal and memory that evolves over conversations.

**Multi-agent simulations.** Social dynamics, trust formation, and belief propagation between agents — the architecture supports multiple independent cognitive brains interacting.

---

## Theoretical Grounding

Cognitiv integrates six established frameworks from cognitive science:

| Framework | Mechanism in Cognitiv |
|-----------|----------------------|
| **ACT-R** (Anderson, 1993) | Base-level activation, spreading activation, fan effect |
| **OCC Model** (Ortony, Clore & Collins, 1988) | Goal/standard/attitude appraisal producing 12 emotion types |
| **Encoding Specificity** (Tulving, 1973) | Contextual reinstatement boost during retrieval |
| **Reconstructive Memory** (Bartlett, 1932) | Interference detection and blended recall |
| **Complementary Learning Systems** (McClelland et al., 1995) | Episodic → semantic consolidation via LLM |
| **Affect Infusion Model** (Forgas, 1995) | Mood-based appraisal bias |

See the [white paper](docs/cognitiv_whitepaper.pdf) for full theoretical discussion and references.

---

## Roadmap

- [x] Core architecture: appraisal, emotion, memory graph, consolidation
- [x] llama.cpp integration
- [x] LLM-assisted appraisal fallback
- [x] Interference detection and blended recall
- [x] Contextual reinstatement and predictive priming
- [ ] Embedding-based seeding (replace keyword matching)
- [ ] Graph visualization tool
- [ ] Cross-agent memory contagion (social memory)
- [ ] Serialization / save-load for long-running simulations
- [ ] C++ port for native Unreal Engine integration
- [ ] User study comparing retrieval architectures on believability

---

## Contributing

Cognitiv is early-stage and feedback is welcome. If you're doing anything interesting with it — research experiments, game integrations, novel extensions — open an issue and let's talk.

Bug reports and PRs should include:
- Python version and platform
- Minimal reproducer for bugs
- Test coverage for new features

---

## Citation

If you use Cognitiv in academic work, please cite:

```bibtex
@software{cognitiv2026,
  title  = {Cognitiv: A Cognitive Architecture for Emotionally Reactive Agents},
  author = {Swamp Rabbit Labs},
  year   = {2026},
  url    = {https://github.com/swamprabbit-labs/cognitiv},
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Cognitiv builds on decades of cognitive science research. Particular thanks to the lineage of cognitive architectures — ACT-R, Soar, ALMA, EMA — whose formal approaches to modeling mind made this possible. And to the authors of *Generative Agents* (Park et al., 2023) for demonstrating that LLM-driven agents with persistent mental state are a tractable research direction.
