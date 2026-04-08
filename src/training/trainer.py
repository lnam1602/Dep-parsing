import torch
import torch.nn.functional as F

from .metrics import attachment_scores


def compute_loss(arc_scores, rel_scores, heads, labels, word_mask):
    arc_logits = arc_scores[:, 1:, :]
    active = word_mask.view(-1)

    flat_arc_logits = arc_logits.reshape(-1, arc_logits.size(-1))[active]
    flat_heads = heads.view(-1)[active]
    arc_loss = F.cross_entropy(flat_arc_logits, flat_heads)

    rel_logits = rel_scores[:, 1:, :, :]
    safe_heads = heads.clamp(min=0)
    gold_head_index = safe_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, rel_logits.size(-1))
    gold_rel_logits = rel_logits.gather(2, gold_head_index).squeeze(2)
    flat_rel_logits = gold_rel_logits.reshape(-1, gold_rel_logits.size(-1))[active]
    flat_labels = labels.view(-1)[active]
    rel_loss = F.cross_entropy(flat_rel_logits, flat_labels)

    return arc_loss + rel_loss


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()

        arc_scores, rel_scores, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_starts"],
            batch["word_mask"],
            upos_ids=batch.get("upos_ids"),
            char_ids=batch.get("char_ids"),
        )
        loss = compute_loss(arc_scores, rel_scores, batch["heads"], batch["labels"], batch["word_mask"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, use_mst: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct_arcs = 0.0
    total_correct_rels = 0.0
    total_tokens = 0.0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        arc_scores, rel_scores, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_starts"],
            batch["word_mask"],
            upos_ids=batch.get("upos_ids"),
            char_ids=batch.get("char_ids"),
        )
        loss = compute_loss(arc_scores, rel_scores, batch["heads"], batch["labels"], batch["word_mask"])
        metrics = attachment_scores(
            arc_scores, rel_scores,
            batch["heads"], batch["labels"], batch["word_mask"],
            use_mst=use_mst,
        )

        token_count = batch["word_mask"].sum().item()
        total_loss += loss.item()
        total_correct_arcs += metrics["uas"] * token_count
        total_correct_rels += metrics["las"] * token_count
        total_tokens += token_count

    if total_tokens == 0:
        return {"loss": 0.0, "uas": 0.0, "las": 0.0}

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "uas": total_correct_arcs / total_tokens,
        "las": total_correct_rels / total_tokens,
    }
