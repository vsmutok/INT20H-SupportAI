import json
import os

from ollama import Client

from src.analyzer.prompts import build_batch_dialog_prompt, build_single_dialog_prompt
from src.config.constants import (
    AGENT_MISTAKES,
    SEED,
    VALID_INTENTS,
)
from src.config.logger import logger


class DatasetAnalyzer:
    def __init__(self, ollama_model: str = "llama3.1:8b", ollama_host: str | None = None):
        host = ollama_host or os.environ.get("OLLAMA_HOST")
        self.client = Client(host=host)
        self.model = ollama_model
        logger.info(f"Initialized DatasetAnalyzer with model: {self.model} (host: {host or 'default'})")

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

    def _format_dialog(self, dialog: list[dict]) -> str:
        """Format dialog messages into a readable string for the LLM."""
        lines = []
        for msg in dialog:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['text']}")
        return "\n".join(lines)

    def analyze_dialog(self, dialog: list[dict]) -> dict:
        """Analyze a single dialog using the LLM. Returns full analysis dict."""
        dialog_text = self._format_dialog(dialog)
        prompt = build_single_dialog_prompt(dialog_text)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0, "seed": SEED},
            )
            result = self._parse_json_response(response["response"])
            return self._validate_analysis(result)
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}. Using defaults.")
            return self._default_analysis()

    def _fuzzy_match_intent(self, intent: str) -> str:
        """Fuzzy-match intent against VALID_INTENTS."""
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

    def _adjust_quality_score(self, score: int, mistakes: list, resolution: str) -> int:
        """Adjust quality score based on mistakes and resolution."""
        adjusted = score - len(mistakes)
        if resolution == "resolved":
            adjusted += 1
        elif resolution == "unresolved":
            adjusted -= 1
        return max(1, min(5, adjusted))

    def _validate_analysis(self, result: dict) -> dict:
        """Validate and normalize LLM analysis output."""
        validated = {}

        validated["intent"] = self._fuzzy_match_intent(result.get("intent", "other"))

        validated["satisfaction"] = self._validate_enum(
            result.get("satisfaction", "neutral"),
            ("satisfied", "neutral", "unsatisfied"),
            "neutral",
        )

        score = self._parse_quality_score(result.get("quality_score", 3))
        validated["quality_score_raw"] = score

        mistakes = result.get("agent_mistakes", [])
        if not isinstance(mistakes, list):
            mistakes = []
        validated["agent_mistakes"] = [m for m in mistakes if m in AGENT_MISTAKES]

        resolution = self._validate_enum(
            result.get("resolution", "unresolved"),
            ("resolved", "partially_resolved", "unresolved"),
            "unresolved",
        )

        validated["quality_score"] = self._adjust_quality_score(score, validated["agent_mistakes"], resolution)
        validated["hidden_dissatisfaction"] = bool(result.get("hidden_dissatisfaction", False))

        validated["tone_agent"] = self._validate_enum(
            result.get("tone_agent", "professional"),
            ("professional", "casual", "rude"),
            "professional",
        )
        validated["tone_client"] = self._validate_enum(
            result.get("tone_client", "calm"),
            ("calm", "frustrated", "angry"),
            "calm",
        )

        validated["resolution"] = resolution
        validated["resolution_summary"] = str(result.get("resolution_summary", ""))[:200]

        return validated

    def _default_analysis(self) -> dict:
        """Return conservative default analysis when LLM fails."""
        return {
            "intent": "other",
            "satisfaction": "neutral",
            "quality_score_raw": 3,
            "quality_score": 3,
            "agent_mistakes": [],
            "hidden_dissatisfaction": False,
            "tone_agent": "professional",
            "tone_client": "calm",
            "resolution": "unresolved",
            "resolution_summary": "Analysis failed — using default values.",
        }

    def analyze_batch(self, dialogs: list[dict], batch_size: int = 5) -> list[dict]:
        """Analyze multiple dialogs in batches for efficiency."""
        results = []
        total = len(dialogs)
        num_batches = (total + batch_size - 1) // batch_size
        logger.info(f"Analyzing {total} dialogs in {num_batches} batches (batch_size={batch_size})...")

        for i in range(0, total, batch_size):
            batch = dialogs[i : i + batch_size]
            batch_num = i // batch_size + 1
            batch_results = self._analyze_batch_prompt(batch)
            results.extend(batch_results)

            done = min(i + batch_size, total)
            pct = done / total * 100
            logger.info(f"  Batch {batch_num}/{num_batches} complete — {done}/{total} dialogs ({pct:.0f}%)")

        logger.success(f"Analysis batch complete — {len(results)} dialogs analyzed")
        return results

    def _match_batch_result(self, parsed: list, idx: int) -> dict | None:
        """Find the matching result for a given index in parsed LLM output."""
        for r in parsed:
            if isinstance(r, dict) and r.get("idx") == idx:
                return r
        if idx < len(parsed) and isinstance(parsed[idx], dict):
            return parsed[idx]
        return None

    def _analyze_batch_prompt(self, batch: list[dict]) -> list[dict]:
        """Analyze a batch of dialogs with a single LLM call."""
        prompt = build_batch_dialog_prompt(batch)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0, "seed": SEED},
            )
            parsed = self._parse_json_response(response["response"])

            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed)}")

            results = []
            for j in range(len(batch)):
                matched = self._match_batch_result(parsed, j)
                if matched:
                    results.append(self._validate_analysis(matched))
                else:
                    results.append(self.analyze_dialog(batch[j].get("dialog", [])))

            return results

        except Exception as e:
            logger.warning(f"Batch analysis failed: {e}. Falling back to individual analysis.")
            return [self.analyze_dialog(d.get("dialog", [])) for d in batch]
