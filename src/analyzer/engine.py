import json
import os
import random
import re

from ollama import Client

from src.analyzer.prompts import (
    MAX_AGENT_CHARS,
    SYSTEM_PROMPT,
    build_single_dialog_prompt,
    build_tier1_prompt,
    build_tier2_prompt,
)
from src.config.constants import (
    AGENT_MISTAKES,
    SEED,
    TEMPLATE_REPLACEMENTS,
    VALID_INTENTS,
)
from src.config.logger import logger


class DatasetAnalyzer:
    def __init__(self, ollama_model: str = "llama3.1:8b", ollama_host: str | None = None):
        host = ollama_host or os.environ.get("OLLAMA_HOST")
        self.client = Client(host=host)
        self.model = ollama_model
        self._rng = random.Random(SEED)
        logger.info(f"Initialized DatasetAnalyzer with model: {self.model} (host: {host or 'default'})")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """Replace {{template}} variables with realistic values."""
        result = text
        for template, replacements in TEMPLATE_REPLACEMENTS.items():
            if template in result:
                value = self._rng.choice(replacements)
                result = result.replace(template, value)
        # Clean up any remaining unreplaced template vars
        result = re.sub(r"\{\{[^}]+\}\}", "", result).strip()
        return result

    def _preprocess_dialog(self, dialog: list[dict]) -> list[dict]:
        """Preprocess all messages in a dialog."""
        return [{"role": msg["role"], "text": self._preprocess_text(msg["text"])} for msg in dialog]

    def _format_dialog(self, dialog: list[dict]) -> str:
        """Format dialog messages into a readable string for the LLM."""
        lines = []
        for msg in dialog:
            role = "C" if msg["role"] == "client" else "A"
            text = msg["text"]
            if role == "A" and len(text) > MAX_AGENT_CHARS:
                text = text[:MAX_AGENT_CHARS] + "..."
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM calls via chat() API
    # ------------------------------------------------------------------

    def _chat(self, user_message: str) -> str:
        """Send a chat message to the LLM with system prompt."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            format="json",
            options={"temperature": 0, "seed": SEED, "num_predict": 2048},
        )
        return response["message"]["content"]

    def _parse_json_response(self, response_text: str):
        """Parse JSON from LLM response, handling markdown fences and extraction."""
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            first_sq = text.find("[")
            first_br = text.find("{")
            candidates = [i for i in [first_sq, first_br] if i != -1]
            start = min(candidates) if candidates else -1
            last_sq = text.rfind("]")
            last_br = text.rfind("}")
            ends = [i for i in [last_sq, last_br] if i != -1]
            end = max(ends) if ends else -1
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    # ------------------------------------------------------------------
    # Validation — pure type/range checks, NO content heuristics
    # ------------------------------------------------------------------

    def _fuzzy_match_intent(self, intent: str) -> str:
        """Normalize LLM intent output to valid intent labels."""
        if intent in VALID_INTENTS:
            return intent
        intent_lower = intent.lower().replace(" ", "_").replace("-", "_")
        for valid in VALID_INTENTS:
            if valid in intent_lower or intent_lower in valid:
                return valid
        return "other"

    def _validate_enum(self, value: str, allowed: tuple, default: str) -> str:
        """Return value if it's in allowed, otherwise return default."""
        return value if value in allowed else default

    def _parse_quality_score(self, raw_score) -> int:
        """Parse and clamp quality score to 1-5."""
        try:
            return max(1, min(5, int(raw_score)))
        except (TypeError, ValueError):
            return 3

    def _validate_tier1(self, result: dict) -> dict:
        """Validate Tier 1 result: intent + satisfaction."""
        return {
            "intent": self._fuzzy_match_intent(result.get("intent", "other")),
            "satisfaction": self._validate_enum(
                result.get("satisfaction", "neutral"),
                ("satisfied", "neutral", "unsatisfied"),
                "neutral",
            ),
        }

    def _validate_tier2(self, result: dict) -> dict:
        """Validate Tier 2 result: quality + mistakes + hidden dissatisfaction."""
        mistakes = result.get("agent_mistakes", [])
        if not isinstance(mistakes, list):
            mistakes = []
        mistakes = [m for m in mistakes if m in AGENT_MISTAKES]

        return {
            "quality_score": self._parse_quality_score(result.get("quality_score", 3)),
            "agent_mistakes": mistakes,
            "hidden_dissatisfaction": bool(result.get("hidden_dissatisfaction", False)),
        }

    def _validate_full(self, result: dict) -> dict:
        """Validate full analysis result (for single-dialog fallback)."""
        t1 = self._validate_tier1(result)
        t2 = self._validate_tier2(result)
        return {**t1, **t2}

    def _default_analysis(self) -> dict:
        """Return conservative default analysis when LLM fails."""
        return {
            "intent": "other",
            "satisfaction": "neutral",
            "quality_score": 3,
            "agent_mistakes": [],
            "hidden_dissatisfaction": False,
        }

    # ------------------------------------------------------------------
    # Single dialog analysis (fallback)
    # ------------------------------------------------------------------

    def analyze_dialog(self, dialog: list[dict]) -> dict:
        """Analyze a single dialog using the LLM. Returns full analysis dict."""
        processed = self._preprocess_dialog(dialog)
        dialog_text = self._format_dialog(processed)
        prompt = build_single_dialog_prompt(dialog_text)

        try:
            response_text = self._chat(prompt)
            result = self._parse_json_response(response_text)
            return self._validate_full(result)
        except Exception as e:
            logger.warning(f"Single dialog analysis failed: {e}. Using defaults.")
            return self._default_analysis()

    # ------------------------------------------------------------------
    # Batch result extraction
    # ------------------------------------------------------------------

    def _extract_batch_results(self, parsed, batch_size: int) -> list[dict | None]:
        """Extract results from parsed LLM output, mapping by idx.

        Returns a list of length batch_size. Missing entries are None.
        """
        results: list[dict | None] = [None] * batch_size

        if isinstance(parsed, dict):
            parsed = parsed["results"] if "results" in parsed and isinstance(parsed["results"], list) else [parsed]

        if not isinstance(parsed, list):
            return results

        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            if isinstance(idx, int) and 0 <= idx < batch_size:
                results[idx] = item

        # If no idx fields, try positional mapping
        if all(r is None for r in results):
            for j, item in enumerate(parsed):
                if j < batch_size and isinstance(item, dict):
                    results[j] = item

        return results

    # ------------------------------------------------------------------
    # Two-tier batch runners
    # ------------------------------------------------------------------

    def _run_tier1_batch(self, batch: list[dict]) -> list[dict | None]:
        """Run Tier 1 (intent + satisfaction) on a preprocessed batch."""
        prompt = build_tier1_prompt(batch)
        try:
            response_text = self._chat(prompt)
            parsed = self._parse_json_response(response_text)
            raw_results = self._extract_batch_results(parsed, len(batch))

            results: list[dict | None] = []
            for r in raw_results:
                if r is not None:
                    results.append(self._validate_tier1(r))
                else:
                    results.append(None)
            return results

        except Exception as e:
            logger.warning(f"Tier 1 batch failed: {e}")
            return [None] * len(batch)

    def _run_tier2_batch(self, batch: list[dict]) -> list[dict | None]:
        """Run Tier 2 (quality + mistakes + HD) on a preprocessed batch."""
        prompt = build_tier2_prompt(batch)
        try:
            response_text = self._chat(prompt)
            parsed = self._parse_json_response(response_text)
            raw_results = self._extract_batch_results(parsed, len(batch))

            results: list[dict | None] = []
            for r in raw_results:
                if r is not None:
                    results.append(self._validate_tier2(r))
                else:
                    results.append(None)
            return results

        except Exception as e:
            logger.warning(f"Tier 2 batch failed: {e}")
            return [None] * len(batch)

    # ------------------------------------------------------------------
    # Tier processing helper
    # ------------------------------------------------------------------

    def _run_tier_batches(
        self,
        preprocessed: list[dict],
        batch_size: int,
        tier_label: str,
        runner,
    ) -> list[dict | None]:
        """Run a tier (1 or 2) over all preprocessed dialogs in batches."""
        total = len(preprocessed)
        num_batches = (total + batch_size - 1) // batch_size
        logger.info(f"{tier_label}: {total} dialogs in {num_batches} batches (size={batch_size})...")
        results: list[dict | None] = [None] * total

        for i in range(0, total, batch_size):
            batch = preprocessed[i : i + batch_size]
            batch_num = i // batch_size + 1
            batch_results = runner(batch)

            for j, result in enumerate(batch_results):
                results[i + j] = result

            done = min(i + batch_size, total)
            pct = done / total * 100
            logger.info(f"  {tier_label[:2]} batch {batch_num}/{num_batches} — {done}/{total} ({pct:.0f}%)")

        return results

    # ------------------------------------------------------------------
    # Merge tiers + fallback
    # ------------------------------------------------------------------

    def _merge_tiers(
        self,
        dialogs: list[dict],
        tier1_results: list[dict | None],
        tier2_results: list[dict | None],
    ) -> list[dict]:
        """Merge tier 1 and tier 2 results, falling back to single-dialog analysis."""
        fallback_count = 0
        final_results = []

        for idx in range(len(dialogs)):
            t1 = tier1_results[idx]
            t2 = tier2_results[idx]

            if t1 is not None and t2 is not None:
                final_results.append({**t1, **t2})
            elif t1 is not None:
                fallback = self.analyze_dialog(dialogs[idx].get("dialog", []))
                final_results.append(
                    {
                        **t1,
                        "quality_score": fallback["quality_score"],
                        "agent_mistakes": fallback["agent_mistakes"],
                        "hidden_dissatisfaction": fallback["hidden_dissatisfaction"],
                    }
                )
                fallback_count += 1
            elif t2 is not None:
                fallback = self.analyze_dialog(dialogs[idx].get("dialog", []))
                final_results.append(
                    {
                        "intent": fallback["intent"],
                        "satisfaction": fallback["satisfaction"],
                        **t2,
                    }
                )
                fallback_count += 1
            else:
                fallback_count += 1
                final_results.append(self.analyze_dialog(dialogs[idx].get("dialog", [])))

        if fallback_count > 0:
            logger.warning(f"Recovered {fallback_count}/{len(dialogs)} dialogs via individual fallback")

        return final_results

    # ------------------------------------------------------------------
    # Main two-tier batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        dialogs: list[dict],
        tier1_batch_size: int = 15,
        tier2_batch_size: int = 10,
    ) -> list[dict]:
        """Analyze multiple dialogs using two-tier batch architecture.

        Tier 1: intent + satisfaction (batch_size=15, fewer output tokens)
        Tier 2: quality + mistakes + HD (batch_size=10, deeper eval)
        Missing entries fall back to individual analysis.
        """
        total = len(dialogs)

        # Preprocess all dialogs
        logger.info(f"Preprocessing {total} dialogs (template replacement)...")
        preprocessed = []
        for d in dialogs:
            processed_dialog = self._preprocess_dialog(d.get("dialog", []))
            preprocessed.append({"dialog": processed_dialog})

        # --- Tier 1: intent + satisfaction ---
        tier1_results = self._run_tier_batches(
            preprocessed, tier1_batch_size, "Tier 1 (intent + satisfaction)", self._run_tier1_batch
        )

        # --- Tier 2: quality + mistakes + HD ---
        tier2_results = self._run_tier_batches(
            preprocessed, tier2_batch_size, "Tier 2 (quality + mistakes + HD)", self._run_tier2_batch
        )

        # --- Merge tiers + fallback for missing ---
        logger.info("Merging tiers and recovering missing entries...")
        final_results = self._merge_tiers(dialogs, tier1_results, tier2_results)

        logger.success(f"Analysis complete — {total} dialogs analyzed")
        return final_results
