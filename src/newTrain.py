"""Train DQU-CIR with an offline Qwen structured-text residual."""

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
    parser.add_argument("--local-rank", "--local_rank", default=int(os.getenv("LOCAL_RANK", -1)), type=int)
    parser.add_argument("--dataset", default="dress", choices=("dress", "shirt", "toptee"))
    parser.add_argument(
        "--fashioniq-split",
        "--fashioniq_split",
        default="original-split",
        choices=("original-split", "val-split"),
    )
    parser.add_argument("--fashioniq-path", "--fashioniq_path", default="../data/FashionIQ")
    parser.add_argument("--structured-train-path")
    parser.add_argument("--structured-val-path")
    parser.add_argument("--structured-only", action="store_true")
    parser.add_argument(
        "--use-written-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep enabled for DQU-CIR; --no-use-written-image is an ablation.",
    )

    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--clip-cache-dir", default="../model_cache")
    parser.add_argument(
        "--freeze-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DQU-CIR fine-tunes CLIP; freezing is only an ablation.",
    )
    parser.add_argument(
        "--dqu-checkpoint",
        help="Optionally initialize shared modules from a trained dquTrain.py checkpoint.",
    )
    parser.add_argument(
        "--freeze-dqu-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Protect the verified DQU checkpoint and optimize only Qwen modules.",
    )
    parser.add_argument(
        "--disable-structured-text",
        "--disable-target-description",
        action="store_true",
        help="Hard-disable the Qwen residual while retaining the DQU-CIR inputs.",
    )

    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--max-structured-weight", type=float, default=0.15)
    parser.add_argument("--max-residual-norm", type=float, default=1.0)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--effective-residual-weight", type=float, default=1.0)
    parser.add_argument("--confidence-calibration-weight", type=float, default=0.2)
    parser.add_argument("--gate-supervision-weight", type=float, default=0.2)
    parser.add_argument("--gate-teacher-temperature", type=float, default=0.1)
    parser.add_argument(
        "--gate-warmup-epochs",
        type=int,
        default=2,
        help="Train only the supervised gate first, then freeze it and fit the adapter.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=int, default=8)
    parser.add_argument("--lr-div", type=float, default=0.1)
    parser.add_argument("--max-decay-epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--model-dir", default="./checkpoints")
    parser.add_argument("--run-name", default="dqu_qwen_structured_text")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-before-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record epoch-0 DQU metrics as the non-degradation checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def default_structured_path(args: argparse.Namespace, split: str) -> str:
    caption_dir = Path(args.fashioniq_path) / "captions"
    candidates = (
        caption_dir / f"structured_edits_{args.dataset}_{split}_qwen37flash.json",
        caption_dir / f"structured_edits_{args.dataset}_{split}.json",
        caption_dir / f"structured_edits_{args.dataset}_{split}_50.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def load_dataset(args: argparse.Namespace, transforms):
    train_path = args.structured_train_path or default_structured_path(args, "train")
    val_path = args.structured_val_path or default_structured_path(args, "val")
    dataset = newDataset.FashionIQ(
        path=args.fashioniq_path,
        category=args.dataset,
        transform=transforms,
        split=args.fashioniq_split,
        structured_train_path=train_path,
        structured_val_path=val_path,
        structured_only=args.structured_only,
        use_written_image=args.use_written_image,
    )
    logging.info(
        "FashionIQ train=%d queries=%d gallery=%d train_structured=%.4f "
        "val_structured=%.4f written_image=%s train_json=%s val_json=%s",
        len(dataset),
        len(dataset.test_queries),
        len(dataset.test_targets),
        dataset.train_structured_coverage,
        dataset.val_structured_coverage,
        args.use_written_image,
        train_path,
        val_path,
    )
    return dataset


def create_model(args, device):
    model = newModel.DQU_CIR(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout_rate,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_checkpoint=args.clip_checkpoint,
        clip_cache_dir=args.clip_cache_dir,
        freeze_clip=args.freeze_clip,
        adapter_rank=args.adapter_rank,
        max_structured_weight=args.max_structured_weight,
        max_residual_norm=args.max_residual_norm,
    ).to(device)
    return model


def create_optimizer(args, model):
    parameters = list(model.named_parameters())
    clip_parameters = [
        parameter
        for name, parameter in parameters
        if name.startswith("clip.") and parameter.requires_grad
    ]
    other_parameters = [
        parameter
        for name, parameter in parameters
        if not name.startswith("clip.") and parameter.requires_grad
    ]
    parameter_groups = []
    if clip_parameters:
        parameter_groups.append({"params": clip_parameters, "lr": args.clip_lr})
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": args.lr})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    logging.info(
        "trainable_parameters=%d total_parameters=%d parameter_groups=%d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters()),
        len(parameter_groups),
    )
    return optimizer


def load_dqu_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in model.state_dict()
        if name.startswith("structured_")
    }
    missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or missing != allowed_missing:
        raise RuntimeError(
            "DQU checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}"
        )
    logging.info(
        "initialized_from_dqu=%s epoch=%s metrics=%s",
        path,
        checkpoint.get("epoch"),
        checkpoint.get("metrics"),
    )


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    if checkpoint.get("adapter_only", False):
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_structured = [
            name
            for name in incompatible.missing_keys
            if name.startswith("structured_")
        ]
        if unexpected or missing_structured:
            raise RuntimeError(
                f"Adapter checkpoint mismatch: missing={missing_structured}, "
                f"unexpected={unexpected}"
            )
    else:
        model.load_state_dict(state, strict=True)
    return checkpoint


def train_one_epoch(
    args, model, optimizer, loader, device, scaler, epoch, phase
):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    running_loss = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"DQU+Qwen {phase} epoch {epoch}")
    for batch_index, data in enumerate(progress):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        target = data["target_img_data"].to(device, non_blocking=True)
        visual = data["visual_query"].to(device, non_blocking=True)
        mask = data["has_structured_text"].to(device)
        confidence = data["structured_confidence"].to(device)
        if args.disable_structured_text:
            mask = torch.zeros_like(mask)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            losses = model.compute_loss(
                data["textual_query"],
                data["structured_text"],
                visual,
                target,
                mask,
                confidence,
                structured_fields=data["structured_fields"],
                structured_field_mask=data["structured_field_mask"],
                structured_quality_features=data["structured_quality_features"],
                preservation_weight=args.preservation_weight,
                gate_supervision_weight=args.gate_supervision_weight,
                gate_teacher_temperature=args.gate_teacher_temperature,
                effective_residual_weight=args.effective_residual_weight,
                confidence_calibration_weight=args.confidence_calibration_weight,
            )
            loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch} batch={batch_index}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip_norm,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach())
        steps += 1
        progress.set_postfix(
            loss=f"{running_loss / steps:.4f}",
            rank=f"{float(losses['ranking'].detach()):.4f}",
            preserve=f"{float(losses['preservation'].detach()):.5f}",
            gate=f"{float(losses['predicted_structured_gate'].detach()):.3f}",
            teacher=f"{float(losses['teacher_gate'].detach()):.3f}",
            residual=f"{float(losses['adapter_residual_norm'].detach()):.4f}",
            effective=f"{float(losses['effective_residual_norm'].detach()):.4f}",
            dqu_text=f"{float(losses['dqu_text_weight'].detach()):.3f}",
        )
    if not steps:
        raise RuntimeError("No training batches were processed")
    return running_loss / steps


def save_checkpoint(args, model, epoch, metrics):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{args.dataset}_{args.run_name}_seed{args.seed}_best.pt"
    adapter_only = bool(args.freeze_dqu_backbone and args.dqu_checkpoint)
    state = model.state_dict()
    if adapter_only:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in state.items()
            if name.startswith("structured_")
        }
    torch.save(
        {
            "epoch": epoch,
            "model_state": state,
            "metrics": dict(metrics),
            "config": vars(args),
            "adapter_only": adapter_only,
            "base_checkpoint": args.dqu_checkpoint,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "selection_metric": "R@10 + R@50",
                **dict(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def train_and_evaluate(args, model, optimizer, dataset, device):
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_score = float("-inf")
    stale_epochs = 0
    if args.eval_before_train:
        metrics = newTest.test(args, model, dataset, args.dataset)
        metric_dict = dict(metrics)
        best_score = (
            metric_dict[f"{args.dataset}_r10"]
            + metric_dict[f"{args.dataset}_r50"]
        )
        path = save_checkpoint(args, model, 0, metrics)
        logging.info(
            "epoch=0 protected_metrics=%s score=%.6f saved=%s",
            metric_dict,
            best_score,
            path,
        )
    for epoch in range(1, args.num_epochs + 1):
        if args.gate_warmup_epochs > 0:
            phase = "gate" if epoch <= args.gate_warmup_epochs else "adapter"
        else:
            phase = "joint"
        model.set_structured_training_phase(phase)
        if epoch == args.gate_warmup_epochs + 1 and phase == "adapter":
            stale_epochs = 0
            logging.info("phase_transition epoch=%d phase=adapter", epoch)
        if epoch > 1 and epoch % args.lr_decay == 0 and epoch <= args.max_decay_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_div
        loss = train_one_epoch(
            args, model, optimizer, loader, device, scaler, epoch, phase
        )
        logging.info(
            "epoch=%d phase=%s train_loss=%.6f", epoch, phase, loss
        )
        if epoch % args.eval_every:
            continue

        metrics = newTest.test(args, model, dataset, args.dataset)
        metric_dict = dict(metrics)
        score = metric_dict[f"{args.dataset}_r10"] + metric_dict[f"{args.dataset}_r50"]
        logging.info("epoch=%d metrics=%s score=%.6f", epoch, metric_dict, score)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            path = save_checkpoint(args, model, epoch, metrics)
            logging.info("saved_best=%s", path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logging.info("early_stop epoch=%d best_score=%.6f", epoch, best_score)
                break


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("ViT-H/14 training requires a CUDA GPU")
    set_seed(args.seed)
    device = torch.device("cuda")
    logging.info("arguments=%s", vars(args))
    model = create_model(args, device)
    if args.dqu_checkpoint:
        load_dqu_checkpoint(args.dqu_checkpoint, model)
    if args.freeze_dqu_backbone:
        if not args.dqu_checkpoint:
            raise ValueError("--freeze-dqu-backbone requires --dqu-checkpoint")
        model.freeze_dqu_backbone()
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model)
        logging.info(
            "loaded_checkpoint=%s epoch=%s", args.resume, checkpoint.get("epoch")
        )
    optimizer = create_optimizer(args, model)
    dataset = load_dataset(
        args, (model.preprocess_train, model.preprocess_val)
    )
    if args.eval_only:
        metrics = newTest.test(args, model, dataset, args.dataset)
        print(json.dumps(dict(metrics), indent=2))
        return
    train_and_evaluate(args, model, optimizer, dataset, device)


if __name__ == "__main__":
    main()
