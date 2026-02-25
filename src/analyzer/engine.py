import json
from loguru import logger
from ollama import Client

from src.config.constants import (
    SEED,
    VALID_INTENTS,
    AGENT_MISTAKES,
)


class DatasetAnalyzer:
    def __init__(self, ollama_model: str = "llama3.1:8b"):
        self.client = Client()
        self.model = ollama_model
        logger.info(f"Initialized DatasetAnalyzer with model: {self.model}")

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

        prompt = f"""You are an expert customer support quality analyst. Analyze this conversation carefully.

CONVERSATION:
{dialog_text}

Return a JSON object with ALL of these fields:

1. "intent": The customer's primary intent. Choose from: {json.dumps(VALID_INTENTS)}

2. "satisfaction": The customer's REAL satisfaction. One of: "satisfied", "neutral", "unsatisfied"
   - "satisfied": Problem clearly resolved, customer is happy
   - "neutral": Unclear outcome or partial resolution
   - "unsatisfied": Problem not resolved, customer frustrated or disappointed

3. "quality_score": Agent performance 1-5
   - 5: Excellent — resolved completely, professional, empathetic
   - 4: Good — mostly resolved, professional
   - 3: Average — attempted to help but incomplete
   - 2: Poor — significant issues
   - 1: Very poor — unhelpful, rude, or harmful

4. "agent_mistakes": List from {json.dumps(AGENT_MISTAKES)} or empty []
   - "ignored_question": Did not address the actual question
   - "incorrect_info": Provided wrong/misleading information
   - "rude_tone": Dismissive, condescending, or impatient
   - "no_resolution": No solution or useful next steps given
   - "unnecessary_escalation": Escalated when could have helped directly

5. "hidden_dissatisfaction": true/false
   True ONLY when ALL conditions met:
   - Customer uses polite/accepting language ("thanks", "okay", "I understand")
   - BUT the actual problem was NOT resolved
   - Customer appears to be giving up rather than genuinely satisfied

6. "tone_agent": Agent's communication tone. One of: "professional", "casual", "rude"

7. "tone_client": Client's emotional tone. One of: "calm", "frustrated", "angry"

8. "resolution": Whether the problem was resolved. One of: "resolved", "partially_resolved", "unresolved"

9. "resolution_summary": One sentence explaining the resolution status

Return ONLY a JSON object."""

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

    def _validate_analysis(self, result: dict) -> dict:
        """Validate and normalize LLM analysis output."""
        validated = {}

        # Intent — fuzzy match
        intent = result.get("intent", "other")
        if intent not in VALID_INTENTS:
            intent_lower = intent.lower().replace(" ", "_").replace("-", "_")
            matched = False
            for valid in VALID_INTENTS:
                if valid in intent_lower or intent_lower in valid:
                    intent = valid
                    matched = True
                    break
            if not matched:
                intent = "other"
        validated["intent"] = intent

        # Satisfaction
        satisfaction = result.get("satisfaction", "neutral")
        if satisfaction not in ("satisfied", "neutral", "unsatisfied"):
            satisfaction = "neutral"
        validated["satisfaction"] = satisfaction

        # Quality score — base from LLM
        score = result.get("quality_score", 3)
        try:
            score = int(score)
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            score = 3
        validated["quality_score_raw"] = score

        # Agent mistakes
        mistakes = result.get("agent_mistakes", [])
        if not isinstance(mistakes, list):
            mistakes = []
        validated["agent_mistakes"] = [m for m in mistakes if m in AGENT_MISTAKES]

        # Resolution
        resolution = result.get("resolution", "unresolved")
        if resolution not in ("resolved", "partially_resolved", "unresolved"):
            resolution = "unresolved"

        # Adjusted quality score: -1 per mistake, +1 if resolved
        adjusted = score
        adjusted -= len(validated["agent_mistakes"])
        if resolution == "resolved":
            adjusted += 1
        elif resolution == "unresolved":
            adjusted -= 1
        validated["quality_score"] = max(1, min(5, adjusted))

        # Hidden dissatisfaction
        validated["hidden_dissatisfaction"] = bool(result.get("hidden_dissatisfaction", False))

        # Tone analysis
        tone_agent = result.get("tone_agent", "professional")
        if tone_agent not in ("professional", "casual", "rude"):
            tone_agent = "professional"
        validated["tone_agent"] = tone_agent

        tone_client = result.get("tone_client", "calm")
        if tone_client not in ("calm", "frustrated", "angry"):
            tone_client = "calm"
        validated["tone_client"] = tone_client

        # Resolution
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
        logger.info(f"Analyzing {total} dialogs in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = dialogs[i : i + batch_size]
            batch_results = self._analyze_batch_prompt(batch)
            results.extend(batch_results)

            done = min(i + batch_size, total)
            if done % 25 == 0 or done >= total:
                logger.info(f"  Progress: {done}/{total}")

        return results

    def _analyze_batch_prompt(self, batch: list[dict]) -> list[dict]:
        """Analyze a batch of dialogs with a single LLM call."""
        prompt = f"""You are an expert customer support quality analyst. Analyze each conversation below.

For each conversation determine ALL of these:
- "intent": one of {json.dumps(VALID_INTENTS)}
- "satisfaction": "satisfied", "neutral", or "unsatisfied"
- "quality_score": 1-5 (5=excellent, 1=very poor)
- "agent_mistakes": list from {json.dumps(AGENT_MISTAKES)} or []
- "hidden_dissatisfaction": true if customer seems polite but problem NOT resolved
- "tone_agent": "professional", "casual", or "rude"
- "tone_client": "calm", "frustrated", or "angry"
- "resolution": "resolved", "partially_resolved", or "unresolved"
- "resolution_summary": one sentence about resolution status

IMPORTANT for hidden_dissatisfaction:
If customer says "okay thanks", "I understand", "fine" but agent did NOT actually solve the problem, set true.

"""
        for j, dialog_data in enumerate(batch):
            dialog_msgs = dialog_data.get("dialog", [])
            prompt += f"\n--- CONVERSATION {j} ---\n"
            for msg in dialog_msgs:
                prompt += f"{msg['role'].capitalize()}: {msg['text'][:300]}\n"

        prompt += f"""
Return a JSON array of exactly {len(batch)} objects. Each must have "idx" (0-based index) plus all fields above.
Return ONLY the JSON array."""

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
                matched = None
                for r in parsed:
                    if isinstance(r, dict) and r.get("idx") == j:
                        matched = r
                        break
                if matched is None and j < len(parsed):
                    matched = parsed[j]

                if matched and isinstance(matched, dict):
                    results.append(self._validate_analysis(matched))
                else:
                    results.append(self.analyze_dialog(batch[j].get("dialog", [])))

            return results

        except Exception as e:
            logger.warning(f"Batch analysis failed: {e}. Falling back to individual analysis.")
            return [self.analyze_dialog(d.get("dialog", [])) for d in batch]
