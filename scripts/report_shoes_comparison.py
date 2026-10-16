"""Read-only experiment audit; write paper/local comparison without starting jobs."""
import argparse
import datetime
import json
import math
from pathlib import Path

PAPER = {
    "title": "Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval",
    "url": "https://arxiv.org/html/2404.15875v1#S4.T2",
    "doi": "10.1145/3626772.3657727",
    "table": 2,
    "random_seeds": 5,
    "mean_percent": {"shoes_r1": 31.47, "shoes_r10": 69.19, "shoes_r50": 88.52},
    "std_percent": {"shoes_r1": 1.31, "shoes_r10": .99, "shoes_r50": .31},
    "average_mean_percent": 63.06,
    "average_std_percent": .69,
    "hardware": "single A100-40G",
    "verification": "Author arXiv manuscript Table 2 and section 4.1.2, checked 2026-10-16",
}
METRICS = ("shoes_r1", "shoes_r10", "shoes_r50")
MODES = ("off", "identity", "annotated")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(values):
    result = {}
    for key in METRICS:
        value = values[key]  # Missing metrics are errors, never zero or a fabricated result.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid {key}")
        result[key] = value
    return result


def build(results):
    results = Path(results).resolve()
    report = read(results / "experiment_report.json")
    if report["config"]["dataset"] != "shoes":
        raise ValueError("Shoes report required")
    seed = report["config"]["seed"]
    stages = {s["stage"]: s["returncode"] for s in report["stages"]}
    base_file = results / "checkpoints" / f"shoes_dqu_original_seed{seed}_best.json"
    base = read(base_file)
    baseline = metrics(base)
    rows = [dict(name="DQU-CIR paper", status="published_reference", metrics=PAPER["mean_percent"],
                 std=PAPER["std_percent"], random_seeds=5),
            dict(name="DQU-CIR local", status="completed_baseline", metrics=baseline, random_seeds=1,
                 epoch=base.get("epoch"), evidence=str(base_file),
                 delta_vs_paper_pp={k: baseline[k] - PAPER["mean_percent"][k] for k in METRICS})]
    for mode in MODES:
        name = f"structured_{mode}"
        path = results / "checkpoints" / f"shoes_{name}_seed{seed}_best.json"
        row = dict(name=name, status="pending", metrics=None, random_seeds=1)
        if path.exists():
            data = read(path)
            values = metrics(data)
            row.update(status="completed" if stages.get(name) == 0 else "intermediate_best",
                       metrics=values, epoch=data.get("epoch"), evidence=str(path),
                       delta_vs_local_baseline_pp={k: values[k] - baseline[k] for k in METRICS})
        elif stages.get(name) == 0:
            raise FileNotFoundError(f"Successful stage lacks checkpoint metrics: {path}")
        rows.append(row)
    generation = {}
    for split, file in report["structured_paths"].items():
        if Path(file).exists():
            payload = read(file)
            generation[split] = {k: payload.get(k) for k in (
                "state", "attempted_queries", "total_queries", "valid_queries", "attempts_complete", "updated_utc", "cost")}
        else:
            generation[split] = dict(state="not_started")
    complete = (report["status"] == "complete"
                and all(stages.get(x) == 0 for x in ("qwen_train", "qwen_val", *(f"structured_{m}" for m in MODES)))
                and all(row["status"] == "completed" for row in rows[2:]))
    return dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), queue_status=report["status"],
                final_comparison_complete=complete, paper=PAPER, rows=rows, generation=generation,
                selection_metric="R@10 + R@50; Avg is descriptive only",
                interpretation="Paper: five-seed mean/std, A100-40G. Local: seed42, server-reported RTX 4080 SUPER, Torch2.9. Improvement uses the same-environment local baseline; cross-environment differences are descriptive, not significance tests. Extra adapter training budget remains a confound.")


def markdown(data):
    lines = ["# Shoes：论文与本机受控对比", "", f"状态：`{data['queue_status']}`；完整对比：`{data['final_comparison_complete']}`。", "",
             "数值为百分比；论文为五个 seed 的均值±标准差，本机为单 seed。", "",
             "| 方法 | 状态 | R@1 | R@10 | R@50 | Avg |", "|---|---|---:|---:|---:|---:|"]
    for row in data["rows"]:
        values = row["metrics"]
        cells = ["—"] * 4
        if values:
            cells = [f"{values[k]:.2f}" + (f" ± {row['std'][k]:.2f}" if "std" in row else "") for k in METRICS]
            cells.append("63.06 ± 0.69" if "std" in row else f"{sum(values.values()) / 3:.2f}")
        lines.append("| " + " | ".join([row["name"], row["status"], *cells]) + " |")
    lines += ["", f"论文：[Table 2]({PAPER['url']})；实验配置：[官方仓库](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR)。", "",
              "改进幅度以相同环境的本机 DQU 为基线，单位为百分点；论文行作为外部参考。单 seed 与论文五 seed 不可用于显著性结论。中间最佳 checkpoint 不等于完成实验。", "",
              "原定三个结构化分支各六轮，冻结同一本机 DQU；保留全部验证查询和图库。额外训练预算尚未由匹配预算的继续训练对照排除。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem for .json and .md")
    args = parser.parse_args()
    data = build(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in ((".json", json.dumps(data, ensure_ascii=False, indent=2) + "\n"), (".md", markdown(data))):
        path = args.output.with_suffix(suffix)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps(dict(status=data["queue_status"], final_comparison_complete=data["final_comparison_complete"])))


if __name__ == "__main__":
    main()
