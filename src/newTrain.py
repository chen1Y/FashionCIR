"""Train the structured-description FashionIQ retrieval gates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import newDataset
import newModel
import newTest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--dataset", default="dress", choices=("dress", "shirt", "toptee"))
    parser.add_argument(
        "--fashioniq_split",
        default="original-split",
        choices=("original-split", "val-split"),
        help="Use original-split for reported metrics; val-split is only for debugging.",
    )
    parser.add_argument("--fashioniq_path", default="../data/FashionIQ")
    parser.add_argument("--structured-train-path")
    parser.add_argument("--structured-val-path")
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="Keep only queries that have a validated structured target description.",
    )
    parser.add_argument(
        "--use-written-image",
        action="store_true",
        help="Reproduce the raw-data text-on-image baseline. Off by default.",
    )

    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument(
        "--clip-checkpoint",
        help="Local OpenCLIP checkpoint. When set, no CLIP download is attempted.",
    )
    parser.add_argument("--clip-cache-dir", default="../model_cache")
    parser.add_argument(
        "--freeze-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze both OpenCLIP towers; recommended for the gate feasibility test.",
    )
    parser.add_argument("--initial-target-weight", type=float, default=0.25)
    parser.add_argument("--initial-image-weight", type=float, default=0.25)
    parser.add_argument(
        "--disable-target-description",
        action="store_true",
        help="Ablation: force the structured-description gate to zero.",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--model-dir", default="./checkpoints")
    parser.add_argument("--run-name", default="structured_gate")
    parser.add_argument("--resume", help="Load a gate checkpoint before training/evaluation.")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def default_structured_path(args: argparse.Namespace, split: str) -> str:
    caption_dir = Path(args.fashioniq_path) / "captions"
    canonical = caption_dir / f"structured_edits_{args.dataset}_{split}.json"
    if canonical.exists() or split != "val":
        return str(canonical)
    limited = caption_dir / f"structured_edits_{args.dataset}_val_50.json"
    return str(limited)


def load_dataset(args: argparse.Namespace, transforms):
    preprocess_train, preprocess_val = transforms
    train_path = args.structured_train_path or default_structured_path(args, "train")
    val_path = args.structured_val_path or default_structured_path(args, "val")
    dataset = newDataset.FashionIQ(
        path=args.fashioniq_path,
        category=args.dataset,
        transform=(preprocess_train, preprocess_val),
        split=args.fashioniq_split,
        structured_train_path=train_path,
        structured_val_path=val_path,
        structured_only=args.structured_only,
        use_written_image=args.use_written_image,
    )
    logging.info(
        "FashionIQ train=%d val_queries=%d gallery=%d train_structured=%.3f "
        "val_structured=%.3f",
        len(dataset),
        len(dataset.test_queries),
        len(dataset.test_targets),
        dataset.train_structured_coverage,
        dataset.val_structured_coverage,
    )
    return dataset


def create_model_and_optimizer(args: argparse.Namespace, device: torch.device):
    model = newModel.DQU_CIR(
        dropout=args.dropout_rate,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
        freeze_clip=args.freeze_clip,
        initial_target_weight=args.initial_target_weight,
        initial_image_weight=args.initial_image_weight,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    logging.info(
        "trainable_parameters=%d total_parameters=%d",
        sum(parameter.numel() for parameter in trainable),
        sum(parameter.numel() for parameter in model.parameters()),
    )
    return model, optimizer


def train_one_epoch(args, model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}")
    for batch_index, data in enumerate(progress):
        if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
            break
        target_img = data["target_img_data"].to(device, non_blocking=True)
        visual_query = data["visual_query"].to(device, non_blocking=True)
        description_mask = data["has_target_description"].to(device)
        if args.disable_target_description:
            description_mask = torch.zeros_like(description_mask)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            losses = model.compute_loss(
                data["textual_query"],
                data["target_description"],
                visual_query,
                target_img,
                description_mask,
            )
            loss = losses["ranking"]
        loss.backward()
        optimizer.step()

        steps += 1
        running_loss += float(loss.detach())
        progress.set_postfix(
            loss=f"{running_loss / steps:.4f}",
            target_gate=f"{float(losses['target_weight']):.3f}",
            image_gate=f"{float(losses['image_weight']):.3f}",
        )
    if steps == 0:
        raise RuntimeError("No training batches were processed")
    return running_loss / steps


def save_checkpoint(args, model, optimizer, epoch, metrics):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.freeze_clip:
        model_state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if not name.startswith("clip.")
        }
    else:
        model_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "metrics": dict(metrics),
        "config": vars(args),
        "clip_weights_included": not args.freeze_clip,
    }
    path = model_dir / f"{args.dataset}_{args.run_name}_best.pt"
    torch.save(checkpoint, path)
    with (model_dir / f"{args.dataset}_{args.run_name}_best.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(dict(metrics), handle, indent=2)
    return path


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint.get("model_state", checkpoint)
    incompatible = model.load_state_dict(model_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_clip = [
        key for key in incompatible.missing_keys if not key.startswith("clip.")
    ]
    if unexpected or missing_non_clip:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_non_clip}, unexpected={unexpected}"
        )
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def train_and_evaluate(args, model, optimizer, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
    )
    best_score = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(args, model, optimizer, loader, device, epoch)
        logging.info("epoch=%d train_loss=%.6f", epoch, loss)
        if epoch % args.eval_every != 0:
            continue

        metrics = newTest.test(args, model, dataset, args.dataset)
        metric_dict = dict(metrics)
        logging.info("epoch=%d metrics=%s", epoch, metric_dict)
        score = metric_dict[f"{args.dataset}_r10"] + metric_dict[f"{args.dataset}_r50"]
        if score > best_score:
            best_score = score
            stale_epochs = 0
            path = save_checkpoint(args, model, optimizer, epoch, metrics)
            logging.info("saved_best=%s score=%.4f", path, score)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logging.info("early_stop epoch=%d best_score=%.4f", epoch, best_score)
                break


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("arguments=%s", vars(args))
    logging.info("device=%s", device)
    model, optimizer = create_model_and_optimizer(args, device)
    dataset = load_dataset(args, (model.preprocess_train, model.preprocess_val))
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, None if args.eval_only else optimizer)
        logging.info("loaded_checkpoint=%s epoch=%s", args.resume, checkpoint.get("epoch"))
    if args.eval_only:
        metrics = newTest.test(args, model, dataset, args.dataset)
        logging.info("eval_metrics=%s", dict(metrics))
        print(json.dumps(dict(metrics), indent=2))
        return
    train_and_evaluate(args, model, optimizer, dataset, device)


if __name__ == "__main__":
    main()
