import json
import random

from src.config.constants import SEED
from src.config.logger import logger
from src.generator.engine import DatasetAugmenter


def main():
    random.seed(SEED)
    logger.info("=" * 60)
    logger.info("Starting dataset generation process...")
    logger.info("=" * 60)
    augmenter = DatasetAugmenter(ollama_model="llama3.1:8b")

    # Step 1: Load base dataset
    logger.info("Step 1/5: Loading base dataset from HuggingFace...")
    base_dialogs = augmenter.load_base_dataset()

    # Step 2: Analyze subset with LLM
    sample_size = min(100, len(base_dialogs))
    logger.info(f"Step 2/5: Analyzing {sample_size} sample dialogs with LLM (out of {len(base_dialogs)} total)...")
    analyzed_sample = augmenter.analyze_dialog_batch(base_dialogs[:sample_size], batch_size=10)

    # Use analysis patterns for the rest
    remaining = len(base_dialogs) - sample_size
    logger.info(f"Step 3/5: Applying statistical patterns to remaining {remaining} dialogs...")
    for dialog in base_dialogs[sample_size:]:
        dialog["metadata"] = {
            "intent": dialog["base_intent"],
            "satisfaction": random.choice(["satisfied", "satisfied", "satisfied", "neutral", "unsatisfied"]),
            "quality_score": random.randint(3, 5),
            "agent_mistakes": [],
            "hidden_dissatisfaction": False,
        }
        analyzed_sample.append(dialog)
    logger.success(f"Patterns applied. Total dialogs with metadata: {len(analyzed_sample)}")

    # Step 4: Generate variations
    logger.info("Step 4/5: Generating phrase variations for augmentation...")
    augmenter.augment_with_variations(analyzed_sample)

    # Step 5: Expand to target size with max diversity
    target_count = 50_000
    logger.info(f"Step 5/5: Expanding dataset to {target_count} dialogs...")
    expanded_dataset = augmenter.expand_dataset(analyzed_sample, target_count=target_count)

    # Clean up internal fields
    for dialog in expanded_dataset:
        dialog.pop("base_intent", None)
        dialog.pop("original_intent", None)

    # Save
    output_path = "data/dataset.json"
    logger.info(f"Saving {len(expanded_dataset)} dialogs to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(expanded_dataset, f, ensure_ascii=False, indent=2)
    logger.success(f"Dataset saved to {output_path}")

    # Stats calculation
    logger.info("Computing dataset statistics...")
    stats = {
        "total": len(expanded_dataset),
        "by_satisfaction": {},
        "by_intent": {},
        "hidden_dissatisfaction": 0,
        "with_mistakes": 0,
        "dialog_length_distribution": {},
        "mistake_types": {},
    }

    for d in expanded_dataset:
        meta = d.get("metadata", {})
        sat = meta.get("satisfaction", "unknown")
        intent = meta.get("intent", "unknown")

        stats["by_satisfaction"][sat] = stats["by_satisfaction"].get(sat, 0) + 1
        stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + 1

        if meta.get("hidden_dissatisfaction"):
            stats["hidden_dissatisfaction"] += 1

        mistakes = meta.get("agent_mistakes", [])
        if mistakes:
            stats["with_mistakes"] += 1
            for m in mistakes:
                stats["mistake_types"][m] = stats["mistake_types"].get(m, 0) + 1

        dialog_len = str(len(d.get("dialog", [])))
        stats["dialog_length_distribution"][dialog_len] = stats["dialog_length_distribution"].get(dialog_len, 0) + 1

    logger.info("-" * 60)
    logger.success("Dataset generation complete!")
    logger.info("-" * 60)
    logger.info(f"  Total dialogs:           {stats['total']}")
    logger.info(f"  By satisfaction:         {stats['by_satisfaction']}")
    logger.info(f"  By intent:               {stats['by_intent']}")
    logger.info(f"  Hidden dissatisfaction:  {stats['hidden_dissatisfaction']}")
    logger.info(f"  With agent mistakes:     {stats['with_mistakes']}")
    logger.info(f"  Mistake types:           {stats['mistake_types']}")
    logger.info(f"  Dialog lengths:          {stats['dialog_length_distribution']}")
    logger.info("-" * 60)

    with open("data/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to data/dataset_stats.json")


if __name__ == "__main__":
    main()
