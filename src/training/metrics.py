import torch

from .mst import mst_decode_batch


def decode_predictions(arc_scores, rel_scores, word_mask=None, use_mst=False):
    """
    Decode arc and relation scores into predicted heads and labels.

    Args:
        arc_scores: (batch, seq_len, seq_len) — position 0 is ROOT.
        rel_scores: (batch, seq_len, seq_len, n_labels).
        word_mask:  (batch, seq_len-1) bool — required when use_mst=True.
        use_mst:    If True, use Chu-Liu/Edmonds MST decoding instead of argmax.

    Returns:
        pred_heads:  (batch, seq_len-1) int64
        pred_labels: (batch, seq_len-1) int64
    """
    if use_mst and word_mask is not None:
        pred_heads = mst_decode_batch(arc_scores, word_mask)
    else:
        pred_heads = arc_scores[:, 1:, :].argmax(dim=-1)

    gathered_rel_scores = rel_scores[:, 1:, :, :].gather(
        2,
        pred_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, rel_scores.size(-1)),
    ).squeeze(2)
    pred_labels = gathered_rel_scores.argmax(dim=-1)

    return pred_heads, pred_labels


def attachment_scores(arc_scores, rel_scores, gold_heads, gold_labels, word_mask, use_mst=False):
    pred_heads, pred_labels = decode_predictions(arc_scores, rel_scores, word_mask, use_mst)
    active = word_mask

    correct_arcs = (pred_heads.eq(gold_heads) & active).sum().item()
    correct_rels = (pred_heads.eq(gold_heads) & pred_labels.eq(gold_labels) & active).sum().item()
    total = active.sum().item()

    if total == 0:
        return {"uas": 0.0, "las": 0.0}

    return {
        "uas": correct_arcs / total,
        "las": correct_rels / total,
    }
