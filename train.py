import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset.conllu_reader import build_label_vocab, read_conllu
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
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.33)
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args()


def build_dataloader(sentences, label2id, model_name, max_len, batch_size, shuffle):
    dataset = DependencyDataset(
        sentences=sentences,
        label2id=label2id,
        model_name=model_name,
        max_len=max_len,
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

    train_loader = build_dataloader(
        train_sentences, label2id, args.model_name, args.max_len, args.batch_size, True
    )
    dev_loader = build_dataloader(
        dev_sentences, label2id, args.model_name, args.max_len, args.batch_size, False
    )
    test_loader = None
    if test_sentences is not None:
        test_loader = build_dataloader(
            test_sentences, label2id, args.model_name, args.max_len, args.batch_size, False
        )

    model = DependencyParser(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        n_labels=len(label2id),
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_las = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_uas={dev_metrics['uas']:.4f} "
            f"dev_las={dev_metrics['las']:.4f}"
        )

        if dev_metrics["las"] > best_las + args.min_delta:
            best_las = dev_metrics["las"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
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

    print(f"Training finished. Best epoch={best_epoch}, best_dev_las={best_las:.4f}")

    if test_loader is not None:
        checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device)
        print(
            f"Test: loss={test_metrics['loss']:.4f} "
            f"uas={test_metrics['uas']:.4f} "
            f"las={test_metrics['las']:.4f}"
        )


if __name__ == "__main__":
    main()
