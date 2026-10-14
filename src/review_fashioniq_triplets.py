"""Offline two-image Qwen audit proposals. Outputs are NOT training exclusions."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from fashioniq_noise import file_sha, object_sha, read_json


def validate_judgment(value):
    kinds={"garment","logo_only","text_only","other","unknown"}
    if not isinstance(value,dict) or any(not isinstance(value.get(k),str) or value[k] not in kinds for k in ("reference_kind","target_kind")):
        raise ValueError("Invalid image kinds")
    if not isinstance(value.get("recommendation"),str) or value["recommendation"] not in {"keep","exclude","uncertain"}:
        raise ValueError("Invalid recommendation")
    if value.get("captions_conflict") is not None and type(value["captions_conflict"]) is not bool:
        raise ValueError("captions_conflict must be boolean or null")
    if "captions_conflict" not in value or not isinstance(value.get("evidence"),str) or not value["evidence"].strip():
        raise ValueError("Missing conflict assessment/evidence")
    judgments=value.get("caption_judgments")
    if not isinstance(judgments,list) or len(judgments)!=2:
        raise ValueError("Expected two per-caption judgments")
    if any(not isinstance(j,dict) or type(j.get("index")) is not int for j in judgments):
        raise ValueError("Invalid caption index")
    if {j["index"] for j in judgments}!={0,1}:
        raise ValueError("Expected caption indices 0 and 1")
    for j in judgments:
        if not isinstance(j.get("status"),str) or j["status"] not in {"satisfied","violated","unknown"} or not isinstance(j.get("evidence"),str) or not j["evidence"].strip():
            raise ValueError("Invalid per-caption status/evidence")
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--model",default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--offline",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--limit",type=int,default=12,help="0 means all requests")
    parser.add_argument("--max-new-tokens",type=int,default=768)
    parser.add_argument("--dry-run",action="store_true",help="Inspect paired-image requests without loading a model")
    parser.add_argument("--attn-implementation",default="sdpa",choices=("sdpa","eager","flash_attention_2"))
    args=parser.parse_args()
    if args.limit<0 or args.max_new_tokens<1:
        parser.error("Invalid limit/token budget")
    requests=[json.loads(line) for line in Path(args.requests).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("split")!="train" for r in requests):
        raise ValueError("This semantic-audit runner accepts TRAIN requests only")
    if args.limit:
        requests=requests[:args.limit]
    config=dict(model=args.model,max_new_tokens=args.max_new_tokens,attn_implementation=args.attn_implementation)
    output=Path(args.output)
    payload=read_json(output) if output.exists() else dict(schema_version=1,kind="model_review_proposals",config=config,samples=[])
    if payload.get("kind")!="model_review_proposals" or payload.get("config")!=config:
        raise ValueError("Existing model audit has a different configuration")
    by_key={r["request_sha256"]:r for r in payload["samples"]}
    pending=[]
    for r in requests:
        if not all(Path(r[k]).is_file() for k in ("reference_path","target_path")):
            raise FileNotFoundError("Both images must exist for semantic review")
        key=object_sha(dict(request=r,reference_sha=file_sha(r["reference_path"]),target_sha=file_sha(r["target_path"])))
        if key not in by_key or by_key[key]["status"]!="ok":
            pending.append((key,r))
    if not pending:
        print("No pending requests")
        return
    if args.dry_run:
        print(json.dumps(dict(selected=len(pending),previews=[dict(source_index=r["source_index"],
            image_order=[r["reference_path"],r["target_path"]],prompt=r["prompt"]) for _,r in pending[:3]]),indent=2))
        return
    import torch
    from PIL import Image
    from generate_structured_edits import load_qwen, extract_json_object
    try:
        processor,model,device=load_qwen(args)
    except (OSError,RuntimeError,ImportError) as exc:
        payload.update(state="model_unavailable",load_error=f"{type(exc).__name__}: {exc}")
        temporary=output.with_suffix(output.suffix+".tmp")
        temporary.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        temporary.replace(output)
        raise SystemExit(f"Model unavailable; no semantic labels produced. See {output}") from exc
    payload.pop("load_error",None)
    payload["state"]="running"
    for key,r in pending:
        record=dict(request_sha256=key,source_index=r["source_index"],row_sha256=r["row_sha256"],
                    dataset_fingerprint=r["dataset_fingerprint"],category=r["category"],split="train",
                    reviewer=f"model:{args.model}",status="error")
        try:
            content=[{"type":"image","image":r["reference_path"]},
                     {"type":"image","image":r["target_path"]},{"type":"text","text":r["prompt"]}]
            text=processor.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
            with ExitStack() as stack:
                images=[stack.enter_context(Image.open(r[k])).convert("RGB") for k in ("reference_path","target_path")]
                inputs=processor(text=[text],images=images,padding=True,return_tensors="pt").to(device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            response=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
            record["raw_response"]=response
            record["judgment"]=validate_judgment(extract_json_object(response))
            record["status"]="ok"
        except (ValueError,RuntimeError,OSError) as exc:
            record["error"]=f"{type(exc).__name__}: {exc}"
        by_key[key]=record
        payload["samples"]=list(by_key.values())
        temporary=output.with_suffix(output.suffix+".tmp")
        temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
        temporary.replace(output)
        print(f"row={r['source_index']} status={record['status']}",flush=True)
    payload["state"]="complete"
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    temporary.replace(output)


if __name__=="__main__":
    main()
