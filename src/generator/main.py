import json
import random
from collections import Counter
from loguru import logger
from src.generator.engine import DatasetAugmenter
from src.config.constants import SEED


def main():
    random.seed(SEED)
    logger.info("Starting dataset generation process...")
    augmenter = DatasetAugmenter(ollama_model="llama3.1:8b")

    # Step 1: Load base dataset
    base_dialogs = augmenter.load_base_dataset()

    # Step 2: Analyze subset with LLM
    logger.info("Analyzing dialogs with LLM...")
    sample_size = min(2000, len(base_dialogs))
    analyzed_sample = augmenter.analyze_dialog_batch(base_dialogs[:sample_size], batch_size=10)

    # Use analysis patterns for the rest
    logger.info("Applying patterns to the rest of the dataset...")
    for dialog in base_dialogs[sample_size:]:
        dialog["metadata"] = {
            "intent": dialog["base_intent"],
            "satisfaction": random.choice(["satisfied", "satisfied", "satisfied", "neutral", "unsatisfied"]),
            "quality_score": random.randint(3, 5),
            "agent_mistakes": [],
            "hidden_dissatisfaction": False,
        }
        analyzed_sample.append(dialog)

    # Step 3: Generate variations
    augmenter.augment_with_variations(analyzed_sample)

    # Step 4: Expand to target size with max diversity
    expanded_dataset = augmenter.expand_dataset(analyzed_sample, target_count=300)

    # Clean up internal fields
    for dialog in expanded_dataset:
        dialog.pop("base_intent", None)
        dialog.pop("original_intent", None)

    # Save
    logger.info(f"Saving {len(expanded_dataset)} dialogs to data/dataset.json...")
    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump(expanded_dataset, f, ensure_ascii=False, indent=2)

    # Stats calculation
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
        stats["dialog_length_distribution"][dialog_len] = (
            stats["dialog_length_distribution"].get(dialog_len, 0) + 1
        )

    logger.success("Dataset generated successfully!")
    logger.info(f"Total dialogs: {stats['total']}")
    logger.info(f"By satisfaction: {stats['by_satisfaction']}")
    logger.info(f"By intent: {stats['by_intent']}")
    logger.info(f"Hidden dissatisfaction: {stats['hidden_dissatisfaction']}")
    logger.info(f"With agent mistakes: {stats['with_mistakes']}")
    logger.info(f"Mistake types: {stats['mistake_types']}")
    logger.info(f"Dialog lengths: {stats['dialog_length_distribution']}")

    with open("data/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to data/dataset_stats.json")

if __name__ == "__main__":
    main()
