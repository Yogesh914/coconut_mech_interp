# Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought

**Authors:** Hanlin Zhu (UC Berkeley), Shibo Hao (UCSD), Zhiting Hu (UCSD), Jiantao Jiao (UC Berkeley), Stuart Russell (UC Berkeley), Yuandong Tian (Meta AI)

**Venue:** NeurIPS 2025 | **arXiv:** 2505.12514v3

**notebook.ipynb** - This file is how the experiments were done and coded. 
---

## Core Contribution

This paper provides a **theoretical explanation** for why continuous chain-of-thought (COCONUT) outperforms discrete chain-of-thought (CoT) on reasoning tasks. The key insight is that continuous thought vectors act as **superposition states** — encoding multiple search frontiers simultaneously — enabling implicit parallel breadth-first search (BFS), whereas discrete CoT must commit to a single token at each step, forcing sequential (and often suboptimal) search.

## Problem Setting: Directed Graph Reachability

Given a directed graph G = (V, E), a root node r, and two candidate destinations c₁ and c₂ (exactly one reachable from r), determine which candidate is reachable. This problem is fundamental and subsumes many practical reasoning tasks (e.g., knowledge graph traversal, Turing machine simulation).

**Prompt format:** Edges are listed as (source, target, \<e\>) triples, followed by a question token \<Q\>, two candidates, a reasoning token \<R\>, and the root node.

## Main Theoretical Result

**Theorem 1:** A **two-layer transformer** with **D steps of continuous thought** can solve directed graph reachability for any n-vertex graph, where D is the graph's diameter.

- **Continuous CoT:** D steps (diameter of the graph, where D < n)
- **Discrete CoT (best known):** O(n²) steps for constant-depth transformers

This represents a significant complexity separation favoring continuous reasoning.

## Mechanism: How Superposition Works

### The Superposition State (Lemma 2)

Each continuous thought vector [tₖ] is a **normalized superposition** of all vertices reachable from r within c steps:

$$[t_c] = \frac{1}{\sqrt{|V_c|}} \sum_{v \in V_c} u_v$$

where Vₖ is the set of reachable vertices at step c and uᵥ are orthonormal token embeddings.

### Two-Layer Transformer Construction

**Layer 1 (Context Setup):** Five attention heads act as "attention choosers" — each edge token \<e\> copies its source node into buffer 1 and target node into buffer 2. This is achieved by constructing query/key matrices that attend to specific relative positions conditioned on the current token type.

**Layer 2 (Node Expansion):** A single attention head performs the BFS expansion. The current thought [tₖ] (a superposition of Vₖ) queries against all edge tokens. Edges whose source nodes are in Vₖ receive high attention, and their target nodes are added to the superposition, yielding Vₖ₊₁.

**MLP (Noise Filtering):** After attention, weights are non-uniform and noise tokens exist. The MLP uses a threshold activation σ(x) = 1{x ≥ ε} to filter out low-signal tokens and equalize weights of genuinely reachable nodes. Layer normalization then produces a clean, normalized superposition.

**Final Prediction:** The answer token \<A\> "measures" the superposition by comparing the signal strength of c₁ and c₂ within the final thought vector. The candidate with the larger inner product is selected.

### Why Discrete CoT Fails

Discrete CoT must **sample** a single token from the superposition at each step (collapsing the state). This forces sequential, depth-first-style search that may require backtracking — yielding O(n²) steps. Continuous CoT avoids this collapse and maintains all search frontiers in parallel.

## Attention Chooser (Key Building Block)

**Lemma 1:** Under sinusoidal positional encoding, for any token \<x\> and relative offset ℓ, there exist query/key matrices such that position i attends almost entirely to position (i − ℓ) when hᵢ = u_\<x\>, and defaults to position 1 (attention sink) otherwise. This construction also extends to **RoPE** (Appendix B.6).

## Experimental Validation

### Setup
- GPT-2-style decoder, 2 layers, d_model = 768, 8 heads, trained from scratch
- ProsQA dataset: graph reachability problems requiring 3–4 reasoning hops
- Multi-stage training curriculum

### Key Results

| Method | Layers | Accuracy |
|--------|--------|----------|
| No CoT | 2 | ~75% |
| Discrete CoT | 2 | ~75% |
| Discrete CoT* | 12 | ~83% |
| **COCONUT** | **2** | **~100%** |

A 2-layer COCONUT model outperforms a 12-layer discrete CoT model.

### Empirical Confirmation of Superposition

1. **Layer 1 attention:** Edge tokens \<e\> attend almost exclusively to their source and target nodes, matching the theoretical construction.

2. **Layer 2 attention:** Continuous thoughts concentrate attention on **reachable** edges (mean score ~2.12 at step 1 vs. ~0.04 for non-reachable), with additional bias toward **frontier** and **optimal** edges.

3. **Representation analysis:** Inner products between continuous thought [tᵢ] and node embeddings confirm that reachable nodes (especially frontier and optimal nodes) have markedly higher similarity scores than non-reachable nodes, validating the superposition mechanism.

4. **Emergence without explicit supervision:** The superposition behavior emerges from training with only optimal-path supervision. An alternative training method (COCONUT-BFS, using random frontier nodes as targets) achieves the same near-perfect accuracy and converges to similar exploration strategies.

### Exploration Priority

The trained model assigns disproportionate attention to optimal edges, resembling a prioritized search. However, even with COCONUT-BFS training (no optimal-path guidance), the model implicitly performs breadth-first expansion — suggesting the superposition mechanism is a natural consequence of the architecture rather than an artifact of the training curriculum.

## Technical Highlights

- **Embedding structure:** d = 3d_TE + d_PE, divided into content, buffer 1, buffer 2, and positional encoding subspaces
- **Embedding dimension:** O(|Voc|) — linear in vocabulary size
- **Positional encoding compatibility:** Works with both sinusoidal and RoPE encodings (unlike prior constructions that require problem-specific encodings)
- **Graph-independent parameters:** The same transformer parameters work for all graphs up to a maximum size

## Limitations and Future Directions

1. **Lower bound gap:** No proven lower bound on discrete CoT steps for graph reachability (i.e., the separation may not be tight)
2. **Training dynamics:** The emergence of BFS-like exploration from deterministic search-trace supervision lacks theoretical explanation
3. **Generalization:** Results are specific to graph reachability; extending to broader reasoning tasks remains open
4. **Practical scalability:** The MLP filter relies on a threshold activation function (indicator function), which differs from standard activations used in practice

## Key Takeaway

Continuous thought enables **reasoning by superposition** — maintaining and expanding multiple hypotheses in parallel within a single vector representation. This is fundamentally more efficient than discrete CoT, which must serialize exploration through token-by-token sampling. The paper bridges the gap between the empirical success of COCONUT and a rigorous theoretical understanding of its mechanism.
