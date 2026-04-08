"""
Chu-Liu/Edmonds algorithm for Maximum Spanning Tree decoding.
Used at inference time to guarantee valid (cycle-free) dependency trees.

Reference: https://en.wikipedia.org/wiki/Edmonds%27_algorithm
"""

import numpy as np
import torch


def chuliu_edmonds(scores: np.ndarray) -> np.ndarray:
    """
    Chu-Liu/Edmonds algorithm for maximum spanning arborescence rooted at 0.

    Args:
        scores: (n, n) float array where scores[dep, head] = arc score.
                Node 0 is ROOT. Diagonal and scores[0, :] are ignored.

    Returns:
        heads: int array of shape (n-1,) where heads[i] = head of word i+1.
               Values in 0..n-1 (0 = ROOT).
    """
    n = scores.shape[0]
    # Disallow self-loops and ROOT as dependent
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    scores[0, :] = -np.inf

    # greedy: each non-root node picks its best incoming arc
    heads = np.argmax(scores[1:], axis=1)  # shape (n-1,), heads[i] = head of node i+1

    # Check for cycles
    cycle = _find_cycle(heads, n)
    if cycle is None:
        return heads

    # Contract the cycle and recurse
    # cycle: list of node indices (1-indexed in full graph)
    cycle_set = set(cycle)
    # Map old nodes to new nodes in contracted graph
    # Cycle nodes all map to one super-node; others keep shifted indices
    cycle_rep = cycle[0]
    old_to_new = {}
    new_idx = 0
    for i in range(n):
        if i == 0 or i not in cycle_set:
            old_to_new[i] = new_idx
            new_idx += 1
    super_node = new_idx  # index of the contracted super-node
    for c in cycle_set:
        old_to_new[c] = super_node
    new_n = super_node + 1

    # Build contracted score matrix
    new_scores = np.full((new_n, new_n), -np.inf)

    # Score of entering the super-node from external node h:
    #   max over c in cycle of [scores[c, h] - scores[c, heads[c-1]]]
    # (we subtract the cycle arc being "broken")
    cycle_arc_scores = {c: scores[c, heads[c - 1]] for c in cycle_set}

    for dep in range(1, n):
        for head in range(n):
            if dep == head:
                continue
            nd = old_to_new[dep]
            nh = old_to_new[head]
            if nd == nh:
                continue  # both in cycle, skip

            s = scores[dep, head]

            if nd == super_node:
                # dep is in cycle; adjust score
                s_adj = s - cycle_arc_scores[dep]
                if s_adj > new_scores[super_node, nh]:
                    new_scores[super_node, nh] = s_adj
            else:
                if s > new_scores[nd, nh]:
                    new_scores[nd, nh] = s

    # Recurse
    contracted_heads = chuliu_edmonds(new_scores)  # shape (new_n-1,)

    # Expand: find which external arc enters the super-node
    # contracted_heads[i] = head of node i+1 in new graph
    # super_node in new graph is at index super_node, so it's
    # contracted_heads[super_node - 1]
    incoming_to_super = contracted_heads[super_node - 1]  # head (in new graph) of super_node

    # Map back: which original node corresponds to incoming_to_super?
    new_to_old = {}
    for o, nw in old_to_new.items():
        if nw != super_node:
            new_to_old[nw] = o
    orig_external_head = new_to_old[incoming_to_super]

    # Find which cycle node is best entered from orig_external_head
    best_cycle_entry = max(
        cycle_set,
        key=lambda c: scores[c, orig_external_head] - cycle_arc_scores[c],
    )

    # Build final heads array in original node space
    result = np.zeros(n - 1, dtype=np.int64)

    # Fill from contracted solution (non-cycle nodes)
    for new_dep_idx, new_h in enumerate(contracted_heads):
        orig_dep = new_to_old.get(new_dep_idx + 1)  # +1 because contracted_heads is 0-indexed
        if orig_dep is None:
            continue  # this maps to super_node, handled separately
        orig_head = new_to_old.get(new_h)
        if orig_head is None:
            # head is super-node; find the best arc from cycle to this dep
            best_head = max(cycle_set, key=lambda c: scores[orig_dep, c])
            orig_head = best_head
        result[orig_dep - 1] = orig_head

    # Fill cycle nodes: keep cycle arcs, except break at best_cycle_entry
    for c in cycle_set:
        if c == best_cycle_entry:
            result[c - 1] = orig_external_head  # incoming external arc
        else:
            result[c - 1] = heads[c - 1]  # keep the cycle arc

    return result


def _find_cycle(heads: np.ndarray, n: int) -> list | None:
    """
    Detect a cycle in head assignments (heads[i] = head of node i+1, 0-indexed).
    Returns a list of 1-indexed node ids forming the cycle, or None.
    """
    # Build adjacency: node -> its head
    visited = [0] * n  # 0=unvisited, 1=in-progress, 2=done
    # Map: node i+1 -> heads[i]
    head_of = [0] * n
    for i, h in enumerate(heads):
        head_of[i + 1] = int(h)

    for start in range(1, n):
        if visited[start] != 0:
            continue
        path = []
        seen_in_path = {}
        node = start
        while node != 0 and visited[node] != 2:
            if visited[node] == 1:
                # cycle detected
                cycle_start = seen_in_path[node]
                return path[cycle_start:]
            visited[node] = 1
            seen_in_path[node] = len(path)
            path.append(node)
            node = head_of[node]

        for p in path:
            visited[p] = 2

    return None


def mst_decode_batch(
    arc_scores: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Decode a batch of arc score matrices using the MST algorithm.

    Args:
        arc_scores: (batch, seq_len, seq_len) float tensor.
                    arc_scores[b, dep, head] = score of arc head→dep.
                    Position 0 is ROOT; arc_scores[b, 0, :] is ignored.
        word_mask:  (batch, seq_len-1) bool tensor — True for real words.

    Returns:
        pred_heads: (batch, seq_len-1) int64 tensor of predicted head indices.
                    Values in 0..seq_len-1 (0 = ROOT).
    """
    batch_size, seq_len, _ = arc_scores.shape
    n_words = seq_len - 1

    scores_np = arc_scores.detach().cpu().float().numpy()
    mask_np = word_mask.detach().cpu().numpy()

    pred_heads = np.zeros((batch_size, n_words), dtype=np.int64)

    for b in range(batch_size):
        sent_len = int(mask_np[b].sum())
        if sent_len == 0:
            continue
        # Sub-matrix for this sentence: (sent_len+1) x (sent_len+1)
        sub = scores_np[b, : sent_len + 1, : sent_len + 1]
        heads = chuliu_edmonds(sub)  # shape (sent_len,)
        pred_heads[b, :sent_len] = heads

    return torch.tensor(pred_heads, dtype=torch.long, device=arc_scores.device)
