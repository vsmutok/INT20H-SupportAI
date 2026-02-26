import json
from pathlib import Path

from src.analyzer.engine import DatasetAnalyzer
from src.config.logger import logger


def main():
    logger.info("=" * 60)
    logger.info("Starting dataset analysis process...")
    logger.info("=" * 60)

    # Load dataset
    dataset_path = Path("data/dataset.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run generate.py first.")
        return

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Step 1/3: Loaded {len(dataset)} dialogs from {dataset_path}")

    # Initialize analyzer
    analyzer = DatasetAnalyzer(ollama_model="llama3.1:8b")

    # Analyze all dialogs
    logger.info(f"Step 2/3: Analyzing {len(dataset)} dialogs with LLM (batch_size=5)...")
    analysis_results = analyzer.analyze_batch(dataset, batch_size=5)

    # Build output: each dialog with its independent analysis
    logger.info("Building analysis output...")
    output = []
    for dialog_data, analysis in zip(dataset, analysis_results, strict=False):
        entry = {
            "id": dialog_data.get("id", "unknown"),
            "dialog": dialog_data["dialog"],
            "analysis": analysis,
        }
        # Keep generator metadata for comparison (if present)
        if "metadata" in dialog_data:
            entry["generator_metadata"] = dialog_data["metadata"]
        output.append(entry)

    # Save analysis results
    output_path = Path("data/analysis.json")
    logger.info(f"Step 3/3: Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success(f"Analysis saved to {output_path}")

    # Compute and save aggregate statistics
    logger.info("Computing aggregate statistics...")
    stats = _compute_stats(output)
    stats_path = Path("data/analysis_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Stats saved to {stats_path}")

    # Log summary
    logger.info("-" * 60)
    logger.success("Analysis complete!")
    logger.info("-" * 60)
    logger.info(f"  Total dialogs analyzed:      {stats['total']}")
    logger.info(f"  By intent:                   {stats['by_intent']}")
    logger.info(f"  By satisfaction:              {stats['by_satisfaction']}")
    logger.info(f"  Average quality score:        {stats['avg_quality_score']:.2f}")
    logger.info(f"  Hidden dissatisfaction found: {stats['hidden_dissatisfaction_count']}")
    logger.info(f"  Dialogs with agent mistakes:  {stats['with_mistakes_count']}")
    logger.info(f"  Most common mistakes:         {stats['mistake_frequency']}")
    logger.info(f"  Agent tone distribution:      {stats['tone_agent_distribution']}")
    logger.info(f"  Client tone distribution:     {stats['tone_client_distribution']}")
    logger.info(f"  Resolution distribution:      {stats['resolution_distribution']}")
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
        "tone_agent_distribution": {},
        "tone_client_distribution": {},
        "resolution_distribution": {},
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

        tone_a = a.get("tone_agent", "unknown")
        stats["tone_agent_distribution"][tone_a] = stats["tone_agent_distribution"].get(tone_a, 0) + 1

        tone_c = a.get("tone_client", "unknown")
        stats["tone_client_distribution"][tone_c] = stats["tone_client_distribution"].get(tone_c, 0) + 1

        res = a.get("resolution", "unknown")
        stats["resolution_distribution"][res] = stats["resolution_distribution"].get(res, 0) + 1

    if output:
        stats["avg_quality_score"] = round(total_score / len(output), 2)

    return stats


def _log_comparison(dataset: list[dict], analysis_results: list[dict]):
    """Compare LLM analysis against generator metadata to show detection accuracy."""
    match_intent = 0
    match_satisfaction = 0
    match_hidden = 0
    total = min(len(dataset), len(analysis_results))

    for i in range(total):
        meta = dataset[i].get("metadata", {})
        analysis = analysis_results[i]

        if meta.get("intent") == analysis.get("intent"):
            match_intent += 1
        if meta.get("satisfaction") == analysis.get("satisfaction"):
            match_satisfaction += 1
        if meta.get("hidden_dissatisfaction") == analysis.get("hidden_dissatisfaction"):
            match_hidden += 1

    logger.info("-" * 60)
    logger.info("Comparison: Generator metadata vs. LLM analysis")
    logger.info("-" * 60)
    logger.info(f"  Intent match:                {match_intent}/{total} ({100 * match_intent / total:.1f}%)")
    logger.info(
        f"  Satisfaction match:           {match_satisfaction}/{total} ({100 * match_satisfaction / total:.1f}%)"
    )
    logger.info(f"  Hidden dissatisfaction match: {match_hidden}/{total} ({100 * match_hidden / total:.1f}%)")
    logger.info("-" * 60)


if __name__ == "__main__":
    main()
