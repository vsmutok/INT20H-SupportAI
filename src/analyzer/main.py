import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.analyzer.engine import DatasetAnalyzer
from src.config.constants import AGENT_MISTAKES, VALID_INTENTS
from src.config.logger import logger


def main():
    logger.info("=" * 60)
    logger.info("Starting dataset analysis process (v6 — disambiguation hints)")
    logger.info("=" * 60)

    # Accept dataset path from command line
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/dataset.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Provide a valid path.")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Step 1/3: Loaded {len(dataset)} dialogs from {dataset_path}")

    # Initialize analyzer
    analyzer = DatasetAnalyzer(ollama_model="llama3.1:8b")

    # Analyze all dialogs
    logger.info(f"Step 2/3: Analyzing {len(dataset)} dialogs with two-tier LLM (v6)...")
    start_time = time.time()
    analysis_results = analyzer.analyze_batch(dataset)
    elapsed = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Build output: each dialog with its independent analysis
    logger.info("Building analysis output...")
    output = []
    for dialog_data, analysis in zip(dataset, analysis_results, strict=False):
        entry = {
            "id": dialog_data.get("id", "unknown"),
            "dialog": dialog_data["dialog"],
            "analysis": analysis,
        }
        if "metadata" in dialog_data:
            entry["generator_metadata"] = dialog_data["metadata"]
        output.append(entry)

    # Save analysis results
    output_path = Path("data/analysis.json")
    logger.info(f"Step 3/3: Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success(f"Analysis saved to {output_path}")

    # Compute and save aggregate statistics
    logger.info("Computing aggregate statistics...")
    stats = _compute_stats(output)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats_path = Path("data/analysis_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Stats saved to {stats_path}")

    # Log summary
    logger.info("-" * 60)
    logger.success("Analysis complete!")
    logger.info("-" * 60)
    logger.info(f"  Total dialogs analyzed:      {stats['total']}")
    logger.info(f"  Time:                        {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"  By intent:                   {stats['by_intent']}")
    logger.info(f"  By satisfaction:              {stats['by_satisfaction']}")
    logger.info(f"  Average quality score:        {stats['avg_quality_score']:.2f}")
    logger.info(f"  Hidden dissatisfaction found: {stats['hidden_dissatisfaction_count']}")
    logger.info(f"  Dialogs with agent mistakes:  {stats['with_mistakes_count']}")
    logger.info(f"  Most common mistakes:         {stats['mistake_frequency']}")
    logger.info("-" * 60)

    # Compare analysis vs generator metadata (if present)
    if dataset and dataset[0].get("metadata"):
        _log_comparison(dataset, analysis_results)


def _compute_stats(output: list[dict]) -> dict:
    """Compute aggregate statistics from analysis results."""
    stats = {
        "total": len(output),
        "by_intent": {},
        "by_satisfaction": {},
        "avg_quality_score": 0.0,
        "quality_score_distribution": {str(i): 0 for i in range(1, 6)},
        "hidden_dissatisfaction_count": 0,
        "with_mistakes_count": 0,
        "mistake_frequency": {},
    }

    total_score = 0
    for entry in output:
        a = entry["analysis"]

        intent = a.get("intent", "unknown")
        stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + 1

        sat = a.get("satisfaction", "unknown")
        stats["by_satisfaction"][sat] = stats["by_satisfaction"].get(sat, 0) + 1

        score = a.get("quality_score", 3)
        total_score += score
        stats["quality_score_distribution"][str(score)] = stats["quality_score_distribution"].get(str(score), 0) + 1

        if a.get("hidden_dissatisfaction"):
            stats["hidden_dissatisfaction_count"] += 1

        mistakes = a.get("agent_mistakes", [])
        if mistakes:
            stats["with_mistakes_count"] += 1
        for m in mistakes:
            stats["mistake_frequency"][m] = stats["mistake_frequency"].get(m, 0) + 1

    if output:
        stats["avg_quality_score"] = round(total_score / len(output), 2)

    return stats


# ---------------------------------------------------------------------------
# Comprehensive comparison — covers ALL 5 task requirements
# ---------------------------------------------------------------------------


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _log_intent_comparison(pairs: list[tuple], total: int):
    """Log intent detection accuracy and confusion matrix."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("1. INTENT DETECTION")
    logger.info("-" * 60)

    correct = sum(1 for m, a in pairs if m.get("intent") == a.get("intent"))
    logger.info(f"  Overall accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    for intent in VALID_INTENTS:
        tp = sum(1 for m, a in pairs if m.get("intent") == intent and a.get("intent") == intent)
        fp = sum(1 for m, a in pairs if m.get("intent") != intent and a.get("intent") == intent)
        fn = sum(1 for m, a in pairs if m.get("intent") == intent and a.get("intent") != intent)
        p, r, f1 = _precision_recall(tp, fp, fn)
        actual = sum(1 for m, _ in pairs if m.get("intent") == intent)
        pred = sum(1 for _, a in pairs if a.get("intent") == intent)
        logger.info(f"  {intent:20s}: P={p:.2f} R={r:.2f} F1={f1:.2f} (actual={actual}, pred={pred})")

    confusion = Counter()
    for m, a in pairs:
        mi, ai = m.get("intent", "?"), a.get("intent", "?")
        if mi != ai:
            confusion[(mi, ai)] += 1
    if confusion:
        logger.info("  Top misclassifications:")
        for (true_i, pred_i), count in confusion.most_common(5):
            logger.info(f"    {true_i} -> {pred_i}: {count}")

    return correct


def _log_satisfaction_comparison(pairs: list[tuple], total: int):
    """Log satisfaction assessment accuracy."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("2. SATISFACTION ASSESSMENT")
    logger.info("-" * 60)

    correct = sum(1 for m, a in pairs if m.get("satisfaction") == a.get("satisfaction"))
    logger.info(f"  Overall accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

    for sat in ("satisfied", "neutral", "unsatisfied"):
        tp = sum(1 for m, a in pairs if m.get("satisfaction") == sat and a.get("satisfaction") == sat)
        fp = sum(1 for m, a in pairs if m.get("satisfaction") != sat and a.get("satisfaction") == sat)
        fn = sum(1 for m, a in pairs if m.get("satisfaction") == sat and a.get("satisfaction") != sat)
        p, r, f1 = _precision_recall(tp, fp, fn)
        actual = sum(1 for m, _ in pairs if m.get("satisfaction") == sat)
        pred = sum(1 for _, a in pairs if a.get("satisfaction") == sat)
        logger.info(f"  {sat:15s}: P={p:.2f} R={r:.2f} F1={f1:.2f} (actual={actual}, pred={pred})")

    sat_confusion = Counter()
    for m, a in pairs:
        ms, as_ = m.get("satisfaction", "?"), a.get("satisfaction", "?")
        if ms != as_:
            sat_confusion[(ms, as_)] += 1
    if sat_confusion:
        logger.info("  Top mismatches:")
        for (true_s, pred_s), count in sat_confusion.most_common(5):
            logger.info(f"    {true_s} -> {pred_s}: {count}")

    return correct


def _log_hd_comparison(pairs: list[tuple], total: int):
    """Log hidden dissatisfaction detection accuracy."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("3. HIDDEN DISSATISFACTION DETECTION")
    logger.info("-" * 60)

    hd_correct = sum(1 for m, a in pairs if m.get("hidden_dissatisfaction") == a.get("hidden_dissatisfaction"))
    logger.info(f"  Overall accuracy: {hd_correct}/{total} ({100 * hd_correct / total:.1f}%)")

    tp = sum(1 for m, a in pairs if m.get("hidden_dissatisfaction") and a.get("hidden_dissatisfaction"))
    fp = sum(1 for m, a in pairs if not m.get("hidden_dissatisfaction") and a.get("hidden_dissatisfaction"))
    fn = sum(1 for m, a in pairs if m.get("hidden_dissatisfaction") and not a.get("hidden_dissatisfaction"))
    p, r, f1 = _precision_recall(tp, fp, fn)

    actual = sum(1 for m, _ in pairs if m.get("hidden_dissatisfaction"))
    pred = sum(1 for _, a in pairs if a.get("hidden_dissatisfaction"))
    logger.info(f"  Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")
    logger.info(f"  True positives: {tp} | False positives: {fp} | False negatives: {fn}")
    logger.info(f"  Actual count: {actual} | Predicted count: {pred}")

    return p, r, f1


def _log_quality_comparison(pairs: list[tuple], total: int):
    """Log quality score assessment accuracy."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("4. QUALITY SCORE ASSESSMENT")
    logger.info("-" * 60)

    exact_match = 0
    close_match = 0
    total_abs_error = 0
    gen_dist = Counter()
    ana_dist = Counter()

    for m, a in pairs:
        gen_score = m.get("quality_score", 3)
        ana_score = a.get("quality_score", 3)
        gen_dist[gen_score] += 1
        ana_dist[ana_score] += 1
        diff = abs(gen_score - ana_score)
        total_abs_error += diff
        if diff == 0:
            exact_match += 1
        if diff <= 1:
            close_match += 1

    mae = total_abs_error / total if total > 0 else 0
    logger.info(f"  Exact match: {exact_match}/{total} ({100 * exact_match / total:.1f}%)")
    logger.info(f"  Close match (+-1): {close_match}/{total} ({100 * close_match / total:.1f}%)")
    logger.info(f"  Mean absolute error: {mae:.2f}")
    logger.info(f"  Generator distribution: {dict(sorted(gen_dist.items()))}")
    logger.info(f"  Analyzer distribution:  {dict(sorted(ana_dist.items()))}")

    return exact_match, close_match, mae


def _log_mistakes_comparison(pairs: list[tuple], total: int):
    """Log agent mistake detection accuracy."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("5. AGENT MISTAKE DETECTION")
    logger.info("-" * 60)

    binary_correct = sum(1 for m, a in pairs if bool(m.get("agent_mistakes")) == bool(a.get("agent_mistakes")))
    logger.info(f"  Has-mistakes binary accuracy: {binary_correct}/{total} ({100 * binary_correct / total:.1f}%)")

    for mistake in AGENT_MISTAKES:
        tp = sum(
            1 for m, a in pairs if mistake in m.get("agent_mistakes", []) and mistake in a.get("agent_mistakes", [])
        )
        fp = sum(
            1 for m, a in pairs if mistake not in m.get("agent_mistakes", []) and mistake in a.get("agent_mistakes", [])
        )
        fn = sum(
            1 for m, a in pairs if mistake in m.get("agent_mistakes", []) and mistake not in a.get("agent_mistakes", [])
        )
        p, r, f1 = _precision_recall(tp, fp, fn)
        actual = sum(1 for m, _ in pairs if mistake in m.get("agent_mistakes", []))
        pred = sum(1 for _, a in pairs if mistake in a.get("agent_mistakes", []))
        logger.info(f"  {mistake:25s}: P={p:.2f} R={r:.2f} F1={f1:.2f} (actual={actual}, pred={pred})")

    jaccard_sum = 0
    for m, a in pairs:
        gen_set = set(m.get("agent_mistakes", []))
        ana_set = set(a.get("agent_mistakes", []))
        union = gen_set | ana_set
        if union:
            jaccard_sum += len(gen_set & ana_set) / len(union)
        else:
            jaccard_sum += 1.0
    avg_jaccard = jaccard_sum / total if total > 0 else 0
    logger.info(f"  Average Jaccard similarity: {avg_jaccard:.2f}")

    return binary_correct, avg_jaccard


def _log_comparison(dataset: list[dict], analysis_results: list[dict]):
    """Compare LLM analysis against generator metadata — all 5 task requirements."""
    total = min(len(dataset), len(analysis_results))
    if total == 0:
        return

    pairs = []
    for i in range(total):
        meta = dataset[i].get("metadata", {})
        analysis = analysis_results[i]
        pairs.append((meta, analysis))

    logger.info("=" * 60)
    logger.info("COMPARISON: Generator metadata vs. Analyzer output")
    logger.info("=" * 60)

    intent_correct = _log_intent_comparison(pairs, total)
    sat_correct = _log_satisfaction_comparison(pairs, total)
    p_hd, r_hd, f1_hd = _log_hd_comparison(pairs, total)
    exact_match, close_match, mae = _log_quality_comparison(pairs, total)
    binary_correct, avg_jaccard = _log_mistakes_comparison(pairs, total)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Intent accuracy:           {100 * intent_correct / total:.1f}%")
    logger.info(f"  Satisfaction accuracy:      {100 * sat_correct / total:.1f}%")
    logger.info(f"  Hidden dissatisfaction F1:  {f1_hd:.2f} (P={p_hd:.2f}, R={r_hd:.2f})")
    exact_pct = 100 * exact_match / total
    close_pct = 100 * close_match / total
    logger.info(f"  Quality score exact:        {exact_pct:.1f}% | +-1: {close_pct:.1f}% | MAE: {mae:.2f}")
    binary_pct = 100 * binary_correct / total
    logger.info(f"  Mistakes binary accuracy:   {binary_pct:.1f}% | Jaccard: {avg_jaccard:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
