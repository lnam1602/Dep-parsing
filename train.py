import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.dataset.char_vocab import build_char_vocab
from src.dataset.conllu_reader import build_label_vocab, build_upos_vocab, read_conllu
from src.dataset.dataset import DependencyDataset, collate_dependency_batch
from src.model.parser import DependencyParser
from src.training.trainer import evaluate, train_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vietnamese dependency parser.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="vinai/phobert-base")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    # Differential learning rates 
    parser.add_argument("--bert-lr", type=float, default=2e-5,
                        help="Learning rate for PhoBERT encoder parameters.")
    parser.add_argument("--head-lr", type=float, default=2e-4,
                        help="Learning rate for task-head parameters (MLP, biaffine, embeddings).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Legacy: set the same LR for all parameters.")
    # MLP sizes 
    parser.add_argument("--arc-hidden-dim", type=int, default=512)
    parser.add_argument("--label-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.33)
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps used for linear LR warmup (0 = no warmup).")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon for cross-entropy loss (0 = disabled).")
    parser.add_argument("--max-grad-norm", type=float, default=5.0,
                        help="Max gradient norm for clipping (0 = disabled).")
    # MST decoding at eval time 
    parser.add_argument("--use-mst", action="store_true", default=False,
                        help="Use Chu-Liu/Edmonds MST decoding during evaluation.")
    # POS-tag embeddings 
    parser.add_argument("--pos-dim", type=int, default=0,
                        help="POS-tag embedding dimension (0 = disabled).")
    # Character CNN embeddings 
    parser.add_argument("--char-embed-dim", type=int, default=50,
                        help="Character embedding dimension for CNN encoder.")
    parser.add_argument("--char-out-dim", type=int, default=0,
                        help="Character CNN output dimension (0 = disabled).")
    parser.add_argument("--max-word-len", type=int, default=30,
                        help="Maximum character sequence length per word.")
    # Visualization / logging
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="TensorBoard log directory. Ví dụ: runs/exp1")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Bật Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="dep-parsing",
                        help="Tên W&B project.")
    return parser.parse_args()


def build_dataloader(sentences, label2id, model_name, max_len, batch_size, shuffle,
                     upos2id=None, char2id=None, max_word_len=30):
    dataset = DependencyDataset(
        sentences=sentences,
        label2id=label2id,
        model_name=model_name,
        max_len=max_len,
        upos2id=upos2id,
        char2id=char2id,
        max_word_len=max_word_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dependency_batch,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sentences = read_conllu(args.train_path)
    dev_sentences = read_conllu(args.dev_path)
    test_sentences = read_conllu(args.test_path) if args.test_path else None

    label2id, id2label = build_label_vocab(train_sentences)

    # POS vocabulary 
    upos2id, id2upos = (None, None)
    if args.pos_dim > 0:
        upos2id, id2upos = build_upos_vocab(train_sentences)

    # Character vocabulary 
    char2id, id2char = (None, None)
    if args.char_out_dim > 0:
        char2id, id2char = build_char_vocab(train_sentences)

    train_loader = build_dataloader(
        train_sentences, label2id, args.model_name, args.max_len, args.batch_size, True,
        upos2id=upos2id, char2id=char2id, max_word_len=args.max_word_len,
    )
    dev_loader = build_dataloader(
        dev_sentences, label2id, args.model_name, args.max_len, args.batch_size, False,
        upos2id=upos2id, char2id=char2id, max_word_len=args.max_word_len,
    )
    test_loader = None
    if test_sentences is not None:
        test_loader = build_dataloader(
            test_sentences, label2id, args.model_name, args.max_len, args.batch_size, False,
            upos2id=upos2id, char2id=char2id, max_word_len=args.max_word_len,
        )

    model = DependencyParser(
        model_name=args.model_name,
        arc_hidden_dim=args.arc_hidden_dim,
        label_hidden_dim=args.label_hidden_dim,
        n_labels=len(label2id),
        dropout=args.dropout,
        n_pos=len(upos2id) if upos2id else 0,
        pos_dim=args.pos_dim,
        n_chars=len(char2id) if char2id else 0,
        char_embed_dim=args.char_embed_dim,
        char_out_dim=args.char_out_dim,
    ).to(device)

    # Differential learning rates 
    if args.lr is not None:
        # Legacy: single LR for everything
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        bert_params = list(model.encoder.parameters())
        bert_param_ids = {id(p) for p in bert_params}
        head_params = [p for p in model.parameters() if id(p) not in bert_param_ids]
        optimizer = AdamW([
            {"params": bert_params, "lr": args.bert_lr},
            {"params": head_params, "lr": args.head_lr},
        ])

    # LR scheduler: linear warmup then linear decay to 0
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    ) if warmup_steps > 0 or total_steps > 0 else None
    print(f"Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")

    # TensorBoard
    writer = None
    if args.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.log_dir))
        print(f"TensorBoard logging to: {args.log_dir}")

    # Weights & Biases
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            print(f"W&B run: {wandb.run.url}")
            use_wandb = True
        except ImportError:
            print("wandb không được cài. Bỏ qua W&B logging. Chạy: pip install wandb")

    best_las = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_dev_loss = float("inf")
    epochs_loss_rising = 0
    loss_patience = max(2, args.patience // 2)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing,
        )
        train_loss = train_metrics["loss"]
        grad_norm  = train_metrics["grad_norm"]
        dev_metrics = evaluate(model, dev_loader, device, use_mst=args.use_mst)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_uas={dev_metrics['uas']:.4f} "
            f"dev_las={dev_metrics['las']:.4f} "
            f"grad_norm={grad_norm:.2f}"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/dev",   dev_metrics["loss"], epoch)
            writer.add_scalar("UAS/dev",    dev_metrics["uas"],  epoch)
            writer.add_scalar("LAS/dev",    dev_metrics["las"],  epoch)
            writer.add_scalar("GradNorm",   grad_norm,           epoch)

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "dev_loss":   dev_metrics["loss"],
                "dev_uas":    dev_metrics["uas"],
                "dev_las":    dev_metrics["las"],
                "grad_norm":  grad_norm,
            }, step=epoch)

        if dev_metrics["las"] > best_las + args.min_delta:
            best_las = dev_metrics["las"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "upos2id": upos2id,
                    "id2upos": id2upos,
                    "char2id": char2id,
                    "id2char": id2char,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Saved new best checkpoint at epoch {epoch} with dev_las={best_las:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch={best_epoch}, best_dev_las={best_las:.4f}"
                )
                break

        # Secondary stop: dev loss diverging while LAS is stagnating
        if dev_metrics["loss"] > best_dev_loss:
            epochs_loss_rising += 1
        else:
            best_dev_loss = dev_metrics["loss"]
            epochs_loss_rising = 0
        if epochs_loss_rising >= loss_patience and epochs_without_improvement >= 1:
            print(
                f"Early stopping at epoch {epoch}: dev loss rose for {epochs_loss_rising} consecutive epochs "
                f"with no LAS improvement. Best epoch={best_epoch}, best_dev_las={best_las:.4f}"
            )
            break

    if writer:
        writer.close()
    if use_wandb:
        wandb.finish()

    print(f"Training finished. Best epoch={best_epoch}, best_dev_las={best_las:.4f}")

    if test_loader is not None:
        checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, use_mst=args.use_mst)
        print(
            f"Test: loss={test_metrics['loss']:.4f} "
            f"uas={test_metrics['uas']:.4f} "
            f"las={test_metrics['las']:.4f}"
        )


if __name__ == "__main__":
    main()
