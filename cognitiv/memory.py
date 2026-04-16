"""
cognitiv.memory — Graph-based memory with spreading activation.

Implements:
- Memory graph with typed nodes and edges
- ACT-R base-level activation with logarithmic decay
- Spreading activation with emotional boost and contextual reinstatement
- Predictive priming from environmental context
- Memory consolidation (episodic clusters → semantic schemas)
- Interference detection and blended recall
- Pruning / forgetting

The graph is stored as adjacency lists in plain dicts — no external
graph libraries required.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .config import MemoryConfig
from .types import (
    ContextSnapshot,
    MemoryEdge,
    MemoryNode,
    RetrievedMemory,
    _new_id,
)


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Type alias for the LLM consolidation callback.
# Signature: (episode_texts: list[str]) -> str
# Returns the schema summary text.
LLMConsolidationCallback = Callable[[list[str]], str]


class MemoryGraph:
    """A graph-based memory system with spreading activation retrieval.

    The graph stores memory nodes connected by typed edges. Retrieval
    uses spreading activation seeded by query-relevant nodes, with
    emotional state and environmental context modulating activation spread.

    Usage:
        graph = MemoryGraph(MemoryConfig())

        # Add memories
        node_id = graph.add_episodic(
            content="Marcus accused me of cheating",
            entity_ids=["marcus"],
            context=ContextSnapshot(location_id="forum"),
            emotional_state={"distress": 0.7, "reproach": 0.5},
            importance=0.8,
            timestamp=100.0,
        )

        # Retrieve with spreading activation
        results = graph.retrieve(
            seed_nodes={"node_abc": 0.8},  # from TF-IDF or keyword match
            current_context=ContextSnapshot(location_id="forum"),
            current_emotions={"distress": 0.3},
            current_time=150.0,
        )
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config

        # Node storage
        self._nodes: dict[str, MemoryNode] = {}

        # Adjacency list: source_id → list of edges
        self._edges: dict[str, list[MemoryEdge]] = defaultdict(list)

        # Reverse adjacency for bidirectional traversal
        self._reverse: dict[str, list[str]] = defaultdict(list)

        # Indexes for fast lookup
        self._entity_index: dict[str, list[str]] = defaultdict(list)
        self._location_index: dict[str, list[str]] = defaultdict(list)
        self._type_index: dict[str, list[str]] = defaultdict(list)

        # Co-activation tracking for consolidation
        self._co_activation_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._retrieval_history: list[list[str]] = []

        # Primed activation (persists between queries)
        self._primed: dict[str, float] = {}

    # ─────────────────────────────────────────
    # Node Creation
    # ─────────────────────────────────────────

    def add_episodic(
        self,
        content: str,
        entity_ids: list[str] | None = None,
        context: ContextSnapshot | None = None,
        emotional_state: dict[str, float] | None = None,
        importance: float = 0.5,
        timestamp: float = 0.0,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create an episodic memory node and wire it into the graph.

        Automatically creates:
        - Entity nodes for any entity_ids not already in the graph
        - Entity_link edges connecting this episode to involved entities
        - Emotional edges connecting to emotion nodes
        - Temporal edge to the most recent episodic node (if any)

        Args:
            content: Natural language description of the event.
            entity_ids: Identifiers of entities involved.
            context: Situational context at the time of the event.
            emotional_state: Agent's emotions at formation time.
            importance: How important this memory is [0, 1].
            timestamp: Simulation time of the event.
            tags: Arbitrary metadata.

        Returns:
            The ID of the new episodic node.
        """
        entity_ids = entity_ids or []
        context = context or ContextSnapshot()
        emotional_state = emotional_state or {}
        tags = tags or {}

        # Apply emotional modulation to importance and decay resistance
        arousal = self._compute_arousal(emotional_state)
        modulated_importance = min(
            importance + arousal * self._config.arousal_importance_weight,
            1.0,
        )
        decay_resistance = arousal * self._config.arousal_decay_resistance_weight

        node = MemoryNode(
            node_type="episodic",
            content=content,
            base_activation=0.5 + modulated_importance * 0.3,
            creation_time=timestamp,
            last_access_time=timestamp,
            access_count=0,
            emotional_snapshot=dict(emotional_state),
            importance=modulated_importance,
            decay_resistance=decay_resistance,
            tags=tags,
        )

        self._insert_node(node)

        # Wire entity edges
        for eid in entity_ids:
            entity_node_id = self._ensure_entity_node(eid)
            self._add_edge(MemoryEdge(
                source_id=node.id,
                target_id=entity_node_id,
                edge_type="entity_link",
                weight=0.5,  # was 0.7 — reduced to prevent activation saturation
                context=context,
                formation_time=timestamp,
                bidirectional=True,
            ))
            self._entity_index[eid].append(node.id)

        # Wire emotional edges
        for etype, intensity in emotional_state.items():
            if intensity >= self._config.emotional_edge_threshold:
                emotion_node_id = self._ensure_emotion_node(etype)
                self._add_edge(MemoryEdge(
                    source_id=node.id,
                    target_id=emotion_node_id,
                    edge_type="emotional",
                    weight=intensity,
                    context=context,
                    formation_time=timestamp,
                    bidirectional=True,
                ))

        # Wire temporal edge to previous episode
        recent_episodes = [
            n for n in self._type_index.get("episodic", [])
            if n != node.id
        ]
        if recent_episodes:
            # Find the most recent by creation time
            prev_id = max(
                recent_episodes,
                key=lambda nid: self._nodes[nid].creation_time,
            )
            self._add_edge(MemoryEdge(
                source_id=prev_id,
                target_id=node.id,
                edge_type="temporal",
                weight=0.25,  # was 0.4 — temporal links shouldn't dominate
                context=context,
                formation_time=timestamp,
                bidirectional=False,
            ))

        # Wire location edge
        if context.location_id:
            self._location_index[context.location_id].append(node.id)

        return node.id

    def add_semantic(
        self,
        content: str,
        source_episode_ids: list[str],
        importance: float = 0.6,
        timestamp: float = 0.0,
    ) -> str:
        """Create a semantic (schema) node from consolidated episodes.

        Args:
            content: The generalized belief or pattern.
            source_episode_ids: The episodic nodes this was derived from.
            importance: Importance of the schema.
            timestamp: When consolidation occurred.

        Returns:
            The ID of the new semantic node.
        """
        # Schema gets boosted activation and high decay resistance
        base_act = 0.0
        source_count = 0
        for eid in source_episode_ids:
            if eid in self._nodes:
                base_act += self._nodes[eid].base_activation
                source_count += 1

        if source_count > 0:
            base_act = (base_act / source_count) * self._config.schema_activation_bonus

        node = MemoryNode(
            node_type="semantic",
            content=content,
            base_activation=min(base_act, 1.0),
            creation_time=timestamp,
            last_access_time=timestamp,
            importance=importance,
            decay_resistance=self._config.schema_decay_resistance,
        )

        self._insert_node(node)

        # Wire hierarchical edges from source episodes
        for eid in source_episode_ids:
            if eid in self._nodes:
                self._add_edge(MemoryEdge(
                    source_id=eid,
                    target_id=node.id,
                    edge_type="hierarchical",
                    weight=0.8,
                    formation_time=timestamp,
                    bidirectional=True,
                ))

                # Mark episode as consolidated
                self._nodes[eid].is_consolidated = True
                self._nodes[eid].consolidated_into = node.id
                self._nodes[eid].decay_resistance = max(
                    0,
                    self._nodes[eid].decay_resistance
                    - self._config.post_consolidation_decay_penalty,
                )

                # Copy entity edges from episodes to schema
                for edge in self._edges.get(eid, []):
                    if edge.edge_type == "entity_link":
                        self._add_edge(MemoryEdge(
                            source_id=node.id,
                            target_id=edge.target_id,
                            edge_type="entity_link",
                            weight=edge.weight * 0.9,
                            formation_time=timestamp,
                            bidirectional=True,
                        ))

        return node.id

    def add_associative_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5,
        timestamp: float = 0.0,
    ) -> None:
        """Manually add an associative edge between two nodes.

        Useful for game logic that knows two memories are related
        but the connection wasn't captured automatically.
        """
        if source_id in self._nodes and target_id in self._nodes:
            self._add_edge(MemoryEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type="associative",
                weight=weight,
                formation_time=timestamp,
                bidirectional=True,
            ))

    # ─────────────────────────────────────────
    # Retrieval (Spreading Activation)
    # ─────────────────────────────────────────

    def retrieve(
        self,
        seed_nodes: dict[str, float],
        current_context: ContextSnapshot | None = None,
        current_emotions: dict[str, float] | None = None,
        current_time: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve memories via spreading activation.

        Args:
            seed_nodes: Initial node activations (e.g., from TF-IDF
                or keyword matching). Maps node_id → activation [0, 1].
            current_context: The agent's current situational context.
                Used for contextual reinstatement (encoding specificity).
            current_emotions: The agent's current emotional state.
                Used for state-dependent retrieval.
            current_time: Current simulation time. Used for computing
                base-level activation (recency and frequency).

        Returns:
            List of RetrievedMemory objects, sorted by activation descending.
        """
        current_context = current_context or ContextSnapshot()
        current_emotions = current_emotions or {}
        cfg = self._config

        # Reset transient activations
        for node in self._nodes.values():
            node.current_activation = 0.0

        # ── Seed Phase ──
        # Seeds use their raw activation value directly.
        # Base-level activation modulates how well targets *receive*
        # activation during spread, not how strongly seeds fire.
        active_set: dict[str, float] = {}

        for node_id, seed_activation in seed_nodes.items():
            if node_id not in self._nodes:
                continue
            node = self._nodes[node_id]
            node.current_activation = seed_activation
            active_set[node_id] = seed_activation

        # Add primed activations
        for node_id, primed_act in self._primed.items():
            if node_id in self._nodes:
                self._nodes[node_id].current_activation += primed_act
                if node_id not in active_set:
                    active_set[node_id] = self._nodes[node_id].current_activation
                else:
                    active_set[node_id] = self._nodes[node_id].current_activation

        # ── Spread Phase ──
        for hop in range(cfg.max_hops):
            next_active: dict[str, float] = {}

            for node_id, activation in active_set.items():
                if activation < cfg.activation_threshold:
                    continue

                node = self._nodes[node_id]
                edges = self._get_outgoing_edges(node_id)

                # Fan factor: prevents highly connected nodes from flooding
                fan_out = len(edges)
                fan_factor = 1.0 / (1.0 + math.log(max(fan_out, 1)))

                for edge in edges:
                    target = self._nodes.get(edge.target_id)
                    if target is None:
                        continue

                    # Compute spread amount
                    spread = (
                        activation
                        * edge.weight
                        * cfg.decay_per_hop
                        * fan_factor
                        * self._emotional_boost(node, current_emotions)
                        * self._context_match(edge.context, current_context)
                    )

                    # Accumulate (multiple paths reinforce)
                    target.current_activation = min(
                        target.current_activation + spread, 1.0
                    )

                    if target.current_activation > cfg.activation_threshold:
                        next_active[edge.target_id] = target.current_activation

            # Budget cap: keep only top N active nodes
            if len(next_active) > cfg.max_active_nodes:
                sorted_nodes = sorted(
                    next_active.items(), key=lambda x: x[1], reverse=True
                )
                next_active = dict(sorted_nodes[: cfg.max_active_nodes])

            if not next_active:
                break

            active_set = next_active

        # ── Retrieval Phase ──
        # Collect all sufficiently activated episodic/semantic nodes
        # Modulate by base-level activation (recency × frequency)
        candidates = []
        for nid, node in self._nodes.items():
            if node.current_activation <= cfg.retrieval_threshold:
                continue
            if node.node_type not in ("episodic", "semantic"):
                continue
            # Base activation blends recency/frequency into the final score
            base = self._compute_base_activation(node, current_time)
            # Weighted blend: 70% spreading activation, 30% base level
            final_activation = node.current_activation * 0.7 + base * 0.3
            node.current_activation = final_activation
            candidates.append((nid, node))

        # Sort by activation
        candidates.sort(key=lambda x: x[1].current_activation, reverse=True)

        # Take top K
        top_k = candidates[: cfg.max_retrieval_results]

        # Update access metadata
        for nid, node in top_k:
            node.last_access_time = current_time
            node.access_count += 1

        # Track co-activations for consolidation
        retrieved_ids = [nid for nid, _ in top_k]
        self._record_co_activations(retrieved_ids)

        # Detect interference and build results
        results = self._build_retrieval_results(top_k)

        return results

    def retrieve_by_entity(
        self,
        entity_id: str,
        current_context: ContextSnapshot | None = None,
        current_emotions: dict[str, float] | None = None,
        current_time: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve all memories related to a specific entity.

        Convenience method that seeds spreading activation from the
        entity node.
        """
        entity_node_id = self._get_entity_node_id(entity_id)
        if entity_node_id is None:
            return []

        return self.retrieve(
            seed_nodes={entity_node_id: 0.8},
            current_context=current_context,
            current_emotions=current_emotions,
            current_time=current_time,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Split text into lowercase words, stripping punctuation."""
        import re
        return set(re.findall(r'[a-z]+', text.lower()))

    def retrieve_by_text(
        self,
        query: str,
        current_context: ContextSnapshot | None = None,
        current_emotions: dict[str, float] | None = None,
        current_time: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve memories by text similarity (simple keyword matching).

        This is the simplest entry point: pass a text query and get
        back relevant memories. For better results, use an external
        TF-IDF or embedding scorer to produce seed_nodes and call
        retrieve() directly.
        """
        query_words = self._tokenize(query)
        stop_words = {"the", "a", "an", "is", "are", "to", "of", "and",
                       "in", "for", "it", "on", "was", "what", "do", "i",
                       "you", "about", "know", "me", "tell", "how", "my",
                       "has", "have", "had", "been", "any", "there", "at",
                       "this", "that", "with", "can", "could", "would",
                       "should", "think", "going", "say"}
        query_words -= stop_words

        if not query_words:
            return []

        seed_nodes: dict[str, float] = {}

        # Phase 1: Entity matching — check if any query words are
        # known entity names. This is the highest-signal match.
        for word in query_words:
            entity_nid = self._get_entity_node_id(word)
            if entity_nid:
                seed_nodes[entity_nid] = 0.9  # strong seed

        # Phase 2: Content keyword matching — find episodic/semantic
        # nodes whose text overlaps with query words.
        for nid, node in self._nodes.items():
            if node.node_type not in ("episodic", "semantic"):
                continue
            content_words = self._tokenize(node.content)
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                # Don't overwrite a stronger entity seed
                if nid not in seed_nodes or seed_nodes[nid] < score:
                    seed_nodes[nid] = min(score, 1.0)

        if not seed_nodes:
            return []

        return self.retrieve(
            seed_nodes=seed_nodes,
            current_context=current_context,
            current_emotions=current_emotions,
            current_time=current_time,
        )

    # ─────────────────────────────────────────
    # Predictive Priming
    # ─────────────────────────────────────────

    def predictive_prime(
        self,
        location_id: str | None = None,
        nearby_entity_ids: list[str] | None = None,
        emotional_state: dict[str, float] | None = None,
    ) -> None:
        """Pre-activate memories based on environmental context.

        Called periodically (every few seconds of simulation time).
        Sets a persistent primed activation that carries over into
        the next retrieval query.
        """
        nearby_entity_ids = nearby_entity_ids or []
        emotional_state = emotional_state or {}
        cfg = self._config

        # Decay existing priming (50% per cycle)
        self._primed = {k: v * 0.5 for k, v in self._primed.items() if v > 0.02}

        # Prime by location
        if location_id:
            for nid in self._location_index.get(location_id, []):
                self._primed[nid] = self._primed.get(nid, 0) + cfg.location_priming_strength

        # Prime by nearby entities
        for eid in nearby_entity_ids:
            for nid in self._entity_index.get(eid, []):
                self._primed[nid] = self._primed.get(nid, 0) + cfg.entity_priming_strength

        # Prime by emotional state
        for etype, intensity in emotional_state.items():
            if intensity < 0.3:
                continue
            emotion_nid = self._get_entity_node_id(f"__emotion_{etype}")
            if emotion_nid:
                for edge in self._edges.get(emotion_nid, []):
                    self._primed[edge.target_id] = (
                        self._primed.get(edge.target_id, 0)
                        + cfg.emotional_priming_strength * intensity
                    )

        # Cap primed activations
        self._primed = {
            k: min(v, 0.4) for k, v in self._primed.items()
        }

    def get_primed_activations(self) -> dict[str, float]:
        """Return current primed activation levels (for debugging)."""
        return dict(self._primed)

    # ─────────────────────────────────────────
    # Consolidation
    # ─────────────────────────────────────────

    def detect_consolidation_clusters(self) -> list[list[str]]:
        """Identify clusters of episodic memories that should be consolidated.

        Uses co-activation frequency: if 3+ episodes have been
        co-retrieved more than `co_activation_threshold` times and
        share at least one entity, they form a cluster.

        Returns:
            List of clusters, where each cluster is a list of node IDs.
        """
        cfg = self._config

        # Build adjacency from co-activation counts
        coact_adj: dict[str, set[str]] = defaultdict(set)
        for (a, b), count in self._co_activation_counts.items():
            if count >= cfg.co_activation_threshold:
                # Check they share an entity
                if self._share_entity(a, b):
                    coact_adj[a].add(b)
                    coact_adj[b].add(a)

        # Find connected components (simple BFS)
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for start in coact_adj:
            if start in visited:
                continue
            # Only consider unconsolidated episodic nodes
            if start not in self._nodes:
                continue
            node = self._nodes[start]
            if node.node_type != "episodic" or node.is_consolidated:
                continue

            component: list[str] = []
            queue = [start]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in self._nodes:
                    continue
                n = self._nodes[current]
                if n.node_type != "episodic" or n.is_consolidated:
                    continue

                visited.add(current)
                component.append(current)
                for neighbor in coact_adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= cfg.min_cluster_size:
                clusters.append(component)

        return clusters

    def consolidate(
        self,
        llm_callback: LLMConsolidationCallback,
        current_time: float = 0.0,
    ) -> list[str]:
        """Run the full consolidation pipeline.

        Detects clusters, generates schema summaries via LLM,
        creates schema nodes, and marks source episodes.

        Args:
            llm_callback: Function that takes a list of episode texts
                and returns a one-sentence generalization.
            current_time: Current simulation time.

        Returns:
            List of newly created schema node IDs.
        """
        clusters = self.detect_consolidation_clusters()
        schema_ids: list[str] = []

        for cluster in clusters:
            episode_texts = [
                self._nodes[nid].content
                for nid in cluster
                if nid in self._nodes
            ]

            if not episode_texts:
                continue

            # Call LLM for schema generation
            schema_text = llm_callback(episode_texts)

            if not schema_text or len(schema_text.strip()) == 0:
                continue

            # Compute importance as max of source episodes
            importance = max(
                self._nodes[nid].importance
                for nid in cluster
                if nid in self._nodes
            )

            schema_id = self.add_semantic(
                content=schema_text.strip(),
                source_episode_ids=cluster,
                importance=importance,
                timestamp=current_time,
            )
            schema_ids.append(schema_id)

        return schema_ids

    # ─────────────────────────────────────────
    # Pruning (Forgetting)
    # ─────────────────────────────────────────

    def prune(self, current_time: float) -> list[str]:
        """Remove decayed, consolidated episodic nodes.

        Only prunes episodes that:
        1. Have been consolidated into a schema
        2. Have base activation below the prune threshold
        3. Haven't been accessed recently

        Returns:
            List of pruned node IDs.
        """
        cfg = self._config
        pruned: list[str] = []

        for nid in list(self._nodes.keys()):
            node = self._nodes[nid]
            if (
                node.is_consolidated
                and node.node_type == "episodic"
                and self._compute_base_activation(node, current_time) < cfg.prune_activation_threshold
                and (current_time - node.last_access_time) > cfg.prune_age_threshold
            ):
                self._remove_node(nid)
                pruned.append(nid)

        return pruned

    # ─────────────────────────────────────────
    # Introspection / Debug
    # ─────────────────────────────────────────

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def get_all_nodes(self, node_type: str | None = None) -> list[MemoryNode]:
        if node_type:
            return [
                self._nodes[nid]
                for nid in self._type_index.get(node_type, [])
                if nid in self._nodes
            ]
        return list(self._nodes.values())

    def get_edges(self, node_id: str) -> list[MemoryEdge]:
        return list(self._edges.get(node_id, []))

    def get_node_count(self) -> int:
        return len(self._nodes)

    def get_edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    def get_stats(self) -> dict:
        """Return summary statistics about the memory graph."""
        type_counts = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type] += 1

        edge_type_counts = defaultdict(int)
        for edges in self._edges.values():
            for edge in edges:
                edge_type_counts[edge.edge_type] += 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self.get_edge_count(),
            "nodes_by_type": dict(type_counts),
            "edges_by_type": dict(edge_type_counts),
            "primed_nodes": len(self._primed),
            "consolidation_clusters": len(self.detect_consolidation_clusters()),
        }

    # ─────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────

    def _insert_node(self, node: MemoryNode) -> None:
        self._nodes[node.id] = node
        self._type_index[node.node_type].append(node.id)

    def _add_edge(self, edge: MemoryEdge) -> None:
        self._edges[edge.source_id].append(edge)
        self._reverse[edge.target_id].append(edge.source_id)

        # If bidirectional, add reverse edge
        if edge.bidirectional:
            reverse_edge = MemoryEdge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                context=edge.context,
                formation_time=edge.formation_time,
                bidirectional=False,  # prevent infinite recursion
            )
            self._edges[edge.target_id].append(reverse_edge)
            self._reverse[edge.source_id].append(edge.target_id)

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges from the graph."""
        if node_id not in self._nodes:
            return

        node = self._nodes[node_id]

        # Remove from edges
        if node_id in self._edges:
            del self._edges[node_id]

        # Remove edges pointing to this node
        for source_id in list(self._reverse.get(node_id, [])):
            if source_id in self._edges:
                self._edges[source_id] = [
                    e for e in self._edges[source_id]
                    if e.target_id != node_id
                ]

        if node_id in self._reverse:
            del self._reverse[node_id]

        # Remove from indexes
        if node.node_type in self._type_index:
            self._type_index[node.node_type] = [
                nid for nid in self._type_index[node.node_type]
                if nid != node_id
            ]

        # Remove from priming
        self._primed.pop(node_id, None)

        del self._nodes[node_id]

    def _ensure_entity_node(self, entity_id: str) -> str:
        """Get or create an entity node for the given ID."""
        canonical_id = f"__entity_{entity_id}"
        if canonical_id not in self._nodes:
            node = MemoryNode(
                id=canonical_id,
                node_type="entity",
                content=entity_id,
                base_activation=0.7,
                importance=0.5,
                decay_resistance=0.9,  # entities decay very slowly
            )
            self._insert_node(node)
        return canonical_id

    def _ensure_emotion_node(self, emotion_type: str) -> str:
        """Get or create an emotion node for the given type."""
        canonical_id = f"__emotion_{emotion_type}"
        if canonical_id not in self._nodes:
            node = MemoryNode(
                id=canonical_id,
                node_type="emotional",
                content=emotion_type,
                base_activation=0.5,
                importance=0.3,
                decay_resistance=0.8,
            )
            self._insert_node(node)
        return canonical_id

    def _get_entity_node_id(self, entity_id: str) -> str | None:
        canonical = f"__entity_{entity_id}"
        return canonical if canonical in self._nodes else None

    def _get_outgoing_edges(self, node_id: str) -> list[MemoryEdge]:
        return self._edges.get(node_id, [])

    def _compute_base_activation(
        self, node: MemoryNode, current_time: float
    ) -> float:
        """ACT-R base-level activation, normalized to [0, 1].

        B = ln(n + 1) - d * ln(t - t_last + 1) + importance * w + resistance

        Where:
        - n = access count (frequency)
        - d = decay rate
        - t - t_last = time since last access (recency)
        - importance * w = importance contribution
        - resistance = decay resistance
        """
        cfg = self._config
        n = node.access_count
        time_since = max(current_time - node.last_access_time, 0.01)

        raw = (
            math.log(n + 1)
            - cfg.base_decay_rate * math.log(time_since + 1)
            + node.importance * cfg.importance_weight
            + node.decay_resistance
        )

        # Normalize via sigmoid to [0, 1]
        return 1.0 / (1.0 + math.exp(-raw))

    def _emotional_boost(
        self, node: MemoryNode, current_emotions: dict[str, float]
    ) -> float:
        """Compute emotional boost for activation spreading.

        Two components:
        1. Formation intensity: memories formed during high emotion spread further.
        2. State match: memories formed in a matching emotional state get a bonus.
        """
        # Formation intensity
        formation_intensity = self._compute_arousal(node.emotional_snapshot)

        # State-dependent match
        state_match = _cosine_similarity(
            node.emotional_snapshot, current_emotions
        )

        return (
            1.0
            + formation_intensity * 0.5
            + state_match * self._config.state_match_weight
        )

    def _context_match(
        self, edge_context: ContextSnapshot, current_context: ContextSnapshot
    ) -> float:
        """Compute contextual reinstatement bonus.

        Returns a multiplier in [1.0, 2.0] based on how well the
        current context matches the context at memory formation.
        """
        cfg = self._config
        score = 0.0

        if (
            edge_context.location_id
            and edge_context.location_id == current_context.location_id
        ):
            score += cfg.context_location_weight

        if (
            edge_context.time_of_day
            and edge_context.time_of_day == current_context.time_of_day
        ):
            score += cfg.context_time_weight

        if (
            edge_context.activity
            and edge_context.activity == current_context.activity
        ):
            score += cfg.context_activity_weight

        # Present entities overlap
        if edge_context.present_entity_ids and current_context.present_entity_ids:
            edge_set = set(edge_context.present_entity_ids)
            curr_set = set(current_context.present_entity_ids)
            union = edge_set | curr_set
            if union:
                overlap = len(edge_set & curr_set) / len(union)
                score += overlap * cfg.context_entities_weight

        # Emotional state overlap
        score += (
            _cosine_similarity(
                edge_context.emotional_state,
                current_context.emotional_state,
            )
            * cfg.context_emotion_weight
        )

        return 1.0 + score  # [1.0, 2.0]

    @staticmethod
    def _compute_arousal(emotional_state: dict[str, float]) -> float:
        """Compute arousal from an emotion dict."""
        if not emotional_state:
            return 0.0
        from .types import HIGH_AROUSAL_EMOTIONS
        high = sum(emotional_state.get(e, 0.0) for e in HIGH_AROUSAL_EMOTIONS)
        total = sum(emotional_state.values())
        if total == 0:
            return 0.0
        return min(high / max(total, 0.01), 1.0)

    def _share_entity(self, node_a_id: str, node_b_id: str) -> bool:
        """Check whether two nodes share at least one entity edge."""
        a_entities = {
            e.target_id for e in self._edges.get(node_a_id, [])
            if e.edge_type == "entity_link"
        }
        b_entities = {
            e.target_id for e in self._edges.get(node_b_id, [])
            if e.edge_type == "entity_link"
        }
        return bool(a_entities & b_entities)

    def _record_co_activations(self, retrieved_ids: list[str]) -> None:
        """Track which nodes are co-activated for consolidation detection."""
        self._retrieval_history.append(retrieved_ids)

        # Keep history bounded
        if len(self._retrieval_history) > 100:
            self._retrieval_history = self._retrieval_history[-50:]

        # Update pairwise counts
        for i, a in enumerate(retrieved_ids):
            for b in retrieved_ids[i + 1:]:
                pair = (min(a, b), max(a, b))
                self._co_activation_counts[pair] += 1

    def _build_retrieval_results(
        self, candidates: list[tuple[str, MemoryNode]]
    ) -> list[RetrievedMemory]:
        """Convert candidates to RetrievedMemory with interference detection."""
        cfg = self._config
        results: list[RetrievedMemory] = []

        for rank, (nid, node) in enumerate(candidates):
            confidence = 1.0
            is_blended = False
            blend_note = ""

            # Check for interference: are there other candidates with
            # similar activation that share entities?
            if node.node_type == "episodic":
                interfering = [
                    (oid, other)
                    for oid, other in candidates
                    if oid != nid
                    and other.node_type == "episodic"
                    and abs(other.current_activation - node.current_activation)
                    < cfg.interference_similarity_threshold * node.current_activation
                    and self._share_entity(nid, oid)
                ]

                if interfering:
                    # This memory is subject to interference
                    confidence = max(0.3, 1.0 - 0.2 * len(interfering))
                    is_blended = True
                    other_contents = [
                        self._nodes[oid].content[:60]
                        for oid, _ in interfering[:2]
                    ]
                    blend_note = (
                        f"May be confused with: {'; '.join(other_contents)}"
                    )

            results.append(RetrievedMemory(
                node=node,
                activation=node.current_activation,
                retrieval_rank=rank,
                confidence=confidence,
                is_blended=is_blended,
                blend_note=blend_note,
            ))

        return results
