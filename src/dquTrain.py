"""Train and evaluate the isolated official DQU-CIR FashionIQ baseline."""

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

import dquDataset
import dquModel
import dquTest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dress", choices=("dress", "shirt", "toptee"))
    parser.add_argument(
        "--fashioniq-split",
        default="original-split",
        choices=("original-split", "val-split"),
    )
    parser.add_argument("--fashioniq-path", default="../data/FashionIQ")
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--clip-cache-dir", default="../model_cache")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-decay", type=int, default=8)
    parser.add_argument("--lr-div", type=float, default=0.1)
    parser.add_argument("--max-decay-epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-dir", default="./checkpoints")
    parser.add_argument("--run-name", default="dqu_official")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke-train-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_model(args, device):
    model = dquModel.DQU_CIR(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout_rate,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
    ).to(device)
    parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in parameters if "clip" in name],
                "lr": args.clip_lr,
            },
            {
                "params": [p for name, p in parameters if "clip" not in name],
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    return model, optimizer


def train_epoch(args, model, optimizer, loader, device, scaler, epoch):
    model.train()
    # This follows the official repository: CLIP is fine-tuned, not frozen.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    running = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"DQU train epoch {epoch}")
    for batch_index, batch in enumerate(progress):
        if args.smoke_train_batches and batch_index >= args.smoke_train_batches:
            break
        target = batch["target_img_data"].to(device, non_blocking=True)
        visual = batch["visual_query"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            loss = model.compute_loss(
                batch["textual_query"], visual, target
            )["ranking"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        steps += 1
        progress.set_postfix(loss=f"{running / steps:.4f}")
    if not steps:
        raise RuntimeError("No DQU-CIR training batches were processed")
    return running / steps


def save_checkpoint(path, args, model, optimizer, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": dict(metrics),
            "config": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The ViT-H/14 DQU-CIR baseline requires a CUDA GPU")
    set_seed(args.seed)
    device = torch.device("cuda")
    model, optimizer = build_model(args, device)
    dataset = dquDataset.FashionIQ(
        path=args.fashioniq_path,
        category=args.dataset,
        transform=(model.preprocess_train, model.preprocess_val),
        split=args.fashioniq_split,
    )
    logging.info(
        "DQU-CIR dataset train=%d queries=%d gallery=%d split=%s",
        len(dataset),
        len(dataset.test_queries),
        len(dataset.test_targets),
        args.fashioniq_split,
    )
    checkpoint_path = (
        Path(args.model_dir)
        / f"{args.dataset}_{args.run_name}_seed{args.seed}_best.pt"
    )
    metrics_path = checkpoint_path.with_suffix(".json")

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if args.eval_only:
        metrics = dquTest.test(args, model, dataset, args.dataset)
        print(json.dumps(dict(metrics), indent=2))
        return

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=loader_generator,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_score = float("-inf")
    stale_epochs = 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        if epoch > 1 and epoch % args.lr_decay == 0 and epoch <= args.max_decay_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_div
        loss = train_epoch(args, model, optimizer, loader, device, scaler, epoch)
        logging.info("epoch=%d train_loss=%.6f", epoch, loss)
        if epoch % args.eval_every:
            continue

        metrics = dquTest.test(args, model, dataset, args.dataset)
        metric_dict = dict(metrics)
        score = metric_dict[f"{args.dataset}_r10"] + metric_dict[f"{args.dataset}_r50"]
        logging.info("epoch=%d metrics=%s score=%.6f", epoch, metric_dict, score)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path, args, model, optimizer, epoch, metrics
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "selection_metric": "R@10 + R@50",
                        **metric_dict,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logging.info("saved best checkpoint to %s", checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logging.info("early stopping after %d stale evaluations", stale_epochs)
                break


if __name__ == "__main__":
    main()
