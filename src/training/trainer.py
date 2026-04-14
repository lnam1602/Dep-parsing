import torch
import torch.nn.functional as F

from .metrics import attachment_scores


def compute_loss(arc_scores, rel_scores, heads, labels, word_mask, label_smoothing=0.0):
    arc_logits = arc_scores[:, 1:, :]
    active = word_mask.view(-1)

    flat_arc_logits = arc_logits.reshape(-1, arc_logits.size(-1))[active]
    flat_heads = heads.view(-1)[active]
    # arc_scores có các vị trí bị mask bằng -1e9 (padding, self-loop).
    # label_smoothing phân phối probability đều lên TẤT CẢ các class, kể cả các vị trí -1e9,
    # dẫn đến -log(softmax(-1e9)) ≈ 1e9 → loss bùng nổ. Không dùng label_smoothing cho arc.
    arc_loss = F.cross_entropy(flat_arc_logits, flat_heads)

    rel_logits = rel_scores[:, 1:, :, :]
    safe_heads = heads.clamp(min=0)
    gold_head_index = safe_heads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, rel_logits.size(-1))
    gold_rel_logits = rel_logits.gather(2, gold_head_index).squeeze(2)
    flat_rel_logits = gold_rel_logits.reshape(-1, gold_rel_logits.size(-1))[active]
    flat_labels = labels.view(-1)[active]
    # rel_loss được gather tại gold head position (không chứa -1e9) → label_smoothing an toàn.
    rel_loss = F.cross_entropy(flat_rel_logits, flat_labels, label_smoothing=label_smoothing)

    return arc_loss + rel_loss


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, dataloader, optimizer, device,
                scheduler=None, max_grad_norm=0.0, label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0

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
        loss = compute_loss(
            arc_scores, rel_scores,
            batch["heads"], batch["labels"], batch["word_mask"],
            label_smoothing=label_smoothing,
        )
        loss.backward()

        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        else:
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_grad_norm += float(grad_norm)

    n = max(len(dataloader), 1)
    return {"loss": total_loss / n, "grad_norm": total_grad_norm / n}


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
