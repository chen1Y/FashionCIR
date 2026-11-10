"""First bounded follow-up: fixed dual, confidence proxy and attribute reranking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.nn.functional as F

import newModel
from newTrain import load_checkpoint, load_dqu_checkpoint
from evaluate_image_keyword_routes import make_dataset, encode_gallery, encode_complete_queries, batches, autocast
from fiq_small_math import blend, adaptive_alpha, ranks_from_scores, block_rerank, metrics
from fashioniq_noise import file_sha
from generate_structured_edits import atomic_write_json
from experiment_storage import require_space

RELATIVE = re.compile(r"\b(more|less|fewer|shorter|longer|wider|narrower|lighter|darker|higher|lower|than|same|similar|not|without|instead)\b", re.I)


def attributes(program, field):
    output = []
    for item in program.get(field, []):
        text = str(item.get("attribute", "")).strip()
        if text and len(text) <= 100 and not RELATIVE.search(text) and text not in output:
            output.append(text)
        if len(output) == 4:
            break
    return output


def make_cache(args):
    main = json.loads((args.results / "experiment_report.json").read_text())
    if main["status"] != "complete":
        raise ValueError("Primary controlled comparison must finish first")
    category = args.dataset
    folder = args.results / category
    base = folder / "checkpoints" / f"{category}_dqu_original_seed42_best.pt"
    adapter = folder / "checkpoints" / f"{category}_structured_off_seed42_best.pt"
    val = folder / f"structured_{category}_val.json"
    evidence = dict(category=category, version="frozen-small-pilot-v1",
        base_sha256=main["checkpoint_hashes"][category], adapter_sha256=file_sha(adapter),
        val_structured_sha256=file_sha(val), primary_protocol_sha256=file_sha(args.results / "protocol.json"),
        model_source_sha256=file_sha(Path(__file__).with_name("newModel.py")),
        pilot_source_sha256=file_sha(__file__), math_source_sha256=file_sha(Path(__file__).with_name('fiq_small_math.py')))
    if file_sha(base) != evidence["base_sha256"]:
        raise ValueError("Protected base checkpoint changed")
    signature = hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    cache_file = args.cache / f"{category}_{signature[:12]}.pt"
    if cache_file.exists():
        cache = torch.load(cache_file,map_location="cpu",weights_only=True)
        if cache["evidence"] != evidence:
            raise ValueError("Stale feature cache")
        return cache
    args.cache.mkdir(parents=True,exist_ok=True)
    require_space(args.cache, 1)
    payload = torch.load(adapter,map_location="cpu",weights_only=True)
    config = payload["config"]
    if config["dataset"] != category:
        raise ValueError("Wrong category adapter")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    model = newModel.DQU_CIR(clip_checkpoint=args.clip_checkpoint, freeze_clip=True,
        adapter_rank=int(config.get("adapter_rank",256)), max_structured_weight=float(config.get("max_structured_weight",.15)),
        max_residual_norm=float(config.get("max_residual_norm",1))).to(device)
    load_dqu_checkpoint(base,model,category)
    load_checkpoint(adapter,model,category)
    model.freeze_dqu_backbone()
    model.eval()
    args.fashioniq_path = str(args.root)
    args.fashioniq_split = "original-split"
    args.structured_train_path = str(folder / f"structured_{category}_train.json")
    args.structured_val_path = str(val)
    dqu = make_dataset(args,model,"dqu")
    qwen = make_dataset(args,model,"qwen-add")
    identities = lambda rows:[(r["source_img_id"],r["target_img_id"],r["textual_query"]) for r in rows]
    if identities(dqu.test_queries) != identities(qwen.test_queries):
        raise ValueError("Written-view queries not aligned")
    expected = {"shirt":(2038,6346),"toptee":(1961,5373)}[category]
    if (len(dqu.test_queries),len(dqu.test_targets)) != expected:
        raise ValueError("Incomplete original validation split or gallery")
    qd = encode_complete_queries(model,dqu.test_queries,16,device)
    qq = encode_complete_queries(model,qwen.test_queries,16,device)
    gallery = encode_gallery(model,dqu.test_targets,16,device)
    index = {r["target_img_id"]:i for i,r in enumerate(dqu.test_targets)}
    targets = torch.tensor([index[r["target_img_id"]] for r in dqu.test_queries])
    sources = torch.tensor([index[r["source_img_id"]] for r in dqu.test_queries])
    valid = torch.tensor([r["has_structured_text"] for r in dqu.test_queries],dtype=torch.bool)
    raw_c = torch.tensor([r["structured_confidence"] for r in dqu.test_queries],device=device)[:,None]
    quality = torch.tensor([r["structured_quality_features"] for r in dqu.test_queries],device=device)
    with torch.inference_mode():
        with autocast(device):
            confidence = model.structured_confidence_calibrator(raw_c,quality).float().cpu().flatten()
    del raw_c, quality
    programs = json.loads(val.read_text())["samples"]
    by_pair = {(r["candidate"],r["target"]):r for r in programs}
    garment = "shirt" if category == "shirt" else "top"
    text_lists = []
    for query in dqu.test_queries:
        sample = by_pair[(query["source_img_id"],query["target_img_id"])]
        edit = sample.get("structured_edit") or {}
        text_lists.append({field:[f"a photo of a {garment} with {text}" for text in attributes(edit,field)]
                           if sample.get("status")=="ok" else [] for field in ("add","remove")})
    unique = sorted({s for row in text_lists for phrases in row.values() for s in phrases})
    encoded = {}
    with torch.inference_mode():
        for texts in batches(unique,16):
            with autocast(device):
                features = F.normalize(model.extract_text_fea(texts),dim=-1).float().cpu()
            encoded.update(zip(texts,features))
    field_features = {}
    for field in ("add","remove"):
        field_features[field] = torch.stack([torch.stack([encoded[s] for s in row[field]]).mean(0)
            if row[field] else torch.zeros(gallery.shape[1]) for row in text_lists])
    cache = dict(evidence=evidence, qd=qd, qq=qq, gallery=gallery, targets=targets, sources=sources,
        confidence=confidence, valid=valid, **field_features,
        eligible_add_queries=sum(bool(r["add"]) for r in text_lists),
        eligible_remove_queries=sum(bool(r["remove"]) for r in text_lists),
        query_ids=identities(dqu.test_queries))
    temporary = cache_file.with_suffix(".partial")
    torch.save(cache,temporary)
    os.replace(temporary,cache_file)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("shirt","toptee"),required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=Path("/dev/shm/fiq_small_features_20261026"))
    parser.add_argument("--clip-checkpoint",required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not repeat a finished pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA encoder required")
    require_space(args.output)
    cache = make_cache(args)
    qd,qq,gallery = (cache[k] for k in ("qd","qq","gallery"))
    targets,sources = cache["targets"],cache["sources"]
    confidence,valid = cache["confidence"],cache["valid"]
    permutation = torch.randperm(len(qd),generator=torch.Generator().manual_seed(42))
    queries = {"v7_off_dqu_written":qd, "qwen_written_only":qq, "fixed_dual_alpha05":blend(qd,qq,.5),
               "bounded_confidence_dual":blend(qd,qq,adaptive_alpha(confidence,valid)),
               "confidence_shuffled_control":blend(qd,qq,adaptive_alpha(confidence[permutation],valid[permutation]))}
    results = {}
    def score(query):
        scores = query @ gallery.T
        scores[torch.arange(len(scores)),sources] = -torch.inf
        return scores
    for name,query in queries.items():
        results[name] = metrics(ranks_from_scores(score(query),targets))
    saved = json.loads((args.results/args.dataset/'checkpoints'/f'{args.dataset}_structured_off_seed42_best.json').read_text())
    reproduction_error = {k:results['v7_off_dqu_written'][k]-saved[args.dataset+'_'+k] for k in ('r1','r10','r50')}
    if any(abs(v) > .0001 for v in reproduction_error.values()):
        raise ValueError('Off-checkpoint reload did not reproduce recorded recall: '+str(reproduction_error))
    base_scores = score(queries["fixed_dual_alpha05"])
    attribute = cache["add"]-.5*cache["remove"]
    for name,vector,conf in [
            ("add_only_rerank",cache["add"],confidence*valid),
            ("add_remove_rerank",attribute,confidence*valid),
            ("attributes_shuffled_control",attribute[permutation],(confidence*valid)[permutation])]:
        attribute_scores = (vector @ gallery.T)*conf[:,None]
        results[name] = metrics(block_rerank(base_scores,attribute_scores,targets,weight=.03,k=100))
    original = results["fixed_dual_alpha05"]
    deltas = {name:{k:values[k]-original[k] for k in ("r1","r10","r50")} for name,values in results.items()}
    payload = dict(category=args.dataset, status="complete", evidence=cache["evidence"], results=results,
        delta_vs_fixed_dual_pp=deltas, queries=len(qd), gallery=len(gallery),
        eligible_add_queries=cache["eligible_add_queries"], eligible_remove_queries=cache["eligible_remove_queries"],
        off_reproduction_error_pp=reproduction_error,
        protocol=dict(seed=42, adapter="primary structured_off", fixed_alpha=.5, adaptive_alpha=".5+.1*(2*calibrated_confidence-1), invalid fallback0",
                      rerank_k=100, rerank_weight=.03, remove_weight=.5, comparative_attributes="excluded", additional_api_calls=0),
        interpretation="Exploratory predeclared frozen-feature hypotheses, all rows reported; no validation-best selection, no trained uncertainty/VQA claim. Shuffled controls included. Strict top100 permutation cannot rescue targets outside100.")
    atomic_write_json(args.output,payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
