"""Read-only FashionIQ paper/server/structured comparison, with explicit pending rows."""
import argparse
import datetime
import json
import math
from pathlib import Path

PAPER = {
    "url": "https://arxiv.org/html/2404.15875v1#S4.T1",
    "table": "Table 1, Original-Split", "seeds": 5, "hardware": "single A100-40G",
    "checked": "2026-10-26",
    "shirt": {"r1": None, "r10": 53.57, "r50": 73.21, "std_r10": .27, "std_r50": .34},
    "toptee": {"r1": None, "r10": 58.48, "r50": 79.23, "std_r10": .46, "std_r50": .29},
}
MODES = ("dqu_original", "structured_off", "structured_identity", "structured_annotated")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_values(data, category):
    values = {k:data[category + "_" + k] for k in ("r1", "r10", "r50")}
    if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
        raise ValueError("Invalid checkpoint metrics")
    return values


def build(results, reloads=None):
    results = Path(results)
    report = read(results / "experiment_report.json")
    stages = {(r["category"], r["stage"]):r for r in report["stages"]}
    categories = {}
    final = report["status"] == "complete"
    for category in ("shirt", "toptee"):
        rows = [dict(method="DQU paper", status="published_reference", values={k:PAPER[category][k] for k in ("r1","r10","r50")},
                     std={"r10":PAPER[category]["std_r10"],"r50":PAPER[category]["std_r50"]})]
        base = None
        for mode in MODES:
            path = results / category / "checkpoints" / f"{category}_{mode}_seed42_best.json"
            stage = stages.get((category,mode),{})
            completed = stage.get("status") == "complete" and stage.get("returncode") == 0
            row = dict(method=mode, status="pending", values=None, delta_vs_server_pp=None)
            if completed and not path.exists():
                raise FileNotFoundError("Completed stage missing metric file: " + str(path))
            if path.exists():
                data = read(path)
                values = metric_values(data, category)
                row.update(status="complete" if completed else "intermediate_best", values=values,
                           epoch=data.get("epoch"), evidence=str(path))
                if mode == "dqu_original":
                    base = values
                    row["delta_vs_paper_pp"] = {k:values[k]-PAPER[category][k] for k in ("r10","r50")}
                elif base is not None:
                    row["delta_vs_server_pp"] = {k:values[k]-base[k] for k in values}
            final &= completed and row["status"] == "complete"
            rows.append(row)
        reload = None
        if reloads is not None:
            file = Path(reloads) / f"{category}.json"
            if file.exists():
                reload = metric_values(read(file), category)
        categories[category] = dict(rows=rows, baseline_reloaded_on_current_server=reload,
            reload_delta_vs_checkpoint_pp={k:reload[k]-base[k] for k in base} if reload is not None and base is not None else None)
        for split in ("train","val"):
            file = results / category / f"structured_{category}_{split}.json"
            payload = read(file) if file.exists() else {}
            categories[category][split + "_generation"] = {k:payload.get(k) for k in ("state","completed_queries","valid_queries")}
            final &= payload.get("state") == "complete"
    return dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), queue_status=report["status"],
                final_comparison_complete=bool(final), paper=PAPER, categories=categories,
                costs=read(results / "cost_summary.json") if (results / "cost_summary.json").exists() else None,
                interpretation="Paper: original-split five-seed mean/std; no R1 reported. Server: seed42. Shirt cloned from earlier server and SHA verified, Toptee trained on current server. Reload evaluations are reported separately. Gains use same-server DQU, not paper means. No significance claim, no mixed three-category average; extra adapter budget is a confound.")


def markdown(data):
    lines = ["# FashionIQ：论文、服务器复现与结构化分支", "", "完整主对照：" + str(data["final_comparison_complete"]), "",
             "论文为Original-Split五seed均值±标准差，未报告R@1；服务器为seed42。所有变化单位为百分点。", ""]
    for category, result in data["categories"].items():
        lines += ["## " + category, "", "| 方法 | 状态 | R@1 | R@10 | R@50 | ΔR@1/10/50 vs服务器DQU |",
                  "|---|---|---:|---:|---:|---|"]
        for row in result["rows"]:
            cells = []
            for k in ("r1","r10","r50"):
                v = row["values"].get(k) if row["values"] is not None else None
                cells.append("—" if v is None else f"{v:.4f}" + (f" ± {row['std'][k]:.2f}" if k in row.get("std",{}) else ""))
            delta = row.get("delta_vs_server_pp")
            lines.append("| " + " | ".join([row["method"],row["status"],*cells,
                          " / ".join(f"{delta[k]:+.4f}" for k in ("r1","r10","r50")) if delta else "—"]) + " |")
        if result["baseline_reloaded_on_current_server"] is not None:
            lines += ["", "当前服务器重载DQU权重评估：" + json.dumps(result["baseline_reloaded_on_current_server"]),
                      "相对保存checkpoint指标差：" + json.dumps(result["reload_delta_vs_checkpoint_pp"])]
        lines += ["", "结构化覆盖：" + json.dumps({s:result[s+"_generation"] for s in ("train","val")},ensure_ascii=False), ""]
    lines += [f"来源：[DQU原论文Table 1]({PAPER['url']})。", "",
              "Shirt基线来自旧机训练并在当前机核验SHA，Toptee在当前机训练；重载评估另列。论文差异仅作外部参考，改进以服务器DQU为基线。中间best不是最终结果，不把缺失值填零，不报告混合协议的三类均值。", "",
              "API费用：" + json.dumps(data["costs"],ensure_ascii=False), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--reloads", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = build(args.results, args.reloads)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix, text in ((".json",json.dumps(data,ensure_ascii=False,indent=2)),(".md",markdown(data))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_name(path.name+".tmp")
        temporary.write_text(text+"\n",encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"final_comparison_complete":data["final_comparison_complete"],"queue_status":data["queue_status"]}))


if __name__ == "__main__":
    main()
