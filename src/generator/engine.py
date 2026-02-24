import json
import random
import hashlib
from loguru import logger
from datasets import load_dataset
from ollama import Client

from src.config.constants import (
    SEED, 
    INTENT_MAP, 
    AGENT_MISTAKES, 
    HIDDEN_DISSATISFACTION_CLOSINGS
)

class DatasetAugmenter:
    def __init__(self, ollama_model: str = "llama3.1:8b"):
        self.client = Client()
        self.model = ollama_model
        self.analysis_cache: dict[str, dict] = {}
        logger.info(f"Initialized DatasetAugmenter with model: {self.model}")

    def load_base_dataset(self) -> list[dict]:
        """Load and transform Bitext dataset"""
        logger.info("Loading Bitext dataset from HuggingFace...")
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

        dialogs = []
        for sample in ds["train"]:
            intent = INTENT_MAP.get(sample.get("intent", ""), "other")

            dialog = {
                "dialog": [
                    {"role": "client", "text": sample["instruction"]},
                    {"role": "agent", "text": sample["response"]},
                ],
                "base_intent": intent,
                "original_intent": sample.get("intent", "unknown"),
            }
            dialogs.append(dialog)

        logger.success(f"Loaded {len(dialogs)} base dialogs")
        return dialogs

    def _parse_json_response(self, response_text: str):
        """Helper to parse JSON from LLM response, handling potential formatting issues"""
        text = response_text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json"):
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:-1])
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the first plausible top-level JSON array or object
            first_sq = text.find('[')
            first_br = text.find('{')
            candidates = [i for i in [first_sq, first_br] if i != -1]
            start_bracket = min(candidates) if candidates else -1

            last_sq = text.rfind(']')
            last_br = text.rfind('}')
            end_candidates = [i for i in [last_sq, last_br] if i != -1]
            end_bracket = max(end_candidates) if end_candidates else -1
            
            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                try:
                    return json.loads(text[start_bracket:end_bracket + 1])
                except json.JSONDecodeError:
                    pass
            raise

    def analyze_dialog_batch(self, dialogs: list[dict], batch_size: int = 10) -> list[dict]:
        """Analyze dialogs in batches via LLM to add quality metrics"""
        analyzed = []
        logger.info(f"Analyzing {len(dialogs)} dialogs in batches of {batch_size}...")

        for i in range(0, len(dialogs), batch_size):
            batch = dialogs[i:i + batch_size]

            prompt = """Analyze these customer support dialogs. For each, provide:
- satisfaction: "satisfied", "neutral", or "unsatisfied" (based on if problem seems resolved)
- quality_score: 1-5 (agent's helpfulness and professionalism)
- agent_mistakes: list from ["ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"] or empty []

Return JSON array with objects: {"idx": 0, "satisfaction": "...", "quality_score": N, "agent_mistakes": [...]}

Dialogs:
"""
            for j, d in enumerate(batch):
                client_msg = d["dialog"][0]["text"]
                agent_msg = d["dialog"][1]["text"]
                prompt += f"\n[{j}] Client: {client_msg[:200]}\nAgent: {agent_msg[:200]}\n"

            prompt += "\nReturn JSON array only."

            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.3, "seed": SEED}
                )

                results = self._parse_json_response(response["response"])

                # Ensure results is a list of dictionaries
                if isinstance(results, dict):
                    results = [results]
                elif not isinstance(results, list):
                    logger.warning(f"Unexpected JSON format from LLM for batch starting at {i}: {type(results)}")
                    results = []

                for result in results:
                    if not isinstance(result, dict):
                        continue
                    idx = result.get("idx", 0)
                    if idx < len(batch):
                        batch[idx]["metadata"] = {
                            "intent": batch[idx]["base_intent"],
                            "satisfaction": result.get("satisfaction", "neutral"),
                            "quality_score": result.get("quality_score", 3),
                            "agent_mistakes": result.get("agent_mistakes", []),
                            "hidden_dissatisfaction": False,
                        }
                        analyzed.append(batch[idx])

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response for batch starting at {i}: {e}. Using fallback values.")
                for d in batch:
                    d["metadata"] = {
                        "intent": d["base_intent"],
                        "satisfaction": random.choice(["satisfied", "satisfied", "neutral", "unsatisfied"]),
                        "quality_score": random.randint(3, 5),
                        "agent_mistakes": [],
                        "hidden_dissatisfaction": False,
                    }
                    analyzed.append(d)

            if (i + batch_size) % 500 == 0:
                logger.info(f"  Analyzed {min(i + batch_size, len(dialogs))}/{len(dialogs)}")

        logger.success(f"Analysis complete. Total analyzed: {len(analyzed)}")
        return analyzed

    def generate_variations(
            self,
            phrase: str,
            variation_type: str,
            count: int = 5
    ) -> list[str]:
        """Generate phrase variations for augmentation"""
        cache_key = f"{variation_type}:{hash(phrase)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        prompt = f"""Generate {count} variations of this {variation_type}. 
Keep the same meaning but vary the wording and tone (polite/neutral/frustrated).
Original: "{phrase}"
Return JSON array of strings only."""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.7, "seed": SEED}
            )
            results = self._parse_json_response(response["response"])
            if isinstance(results, list):
                variations = [str(v) for v in results]
            elif isinstance(results, str):
                variations = [results]
            else:
                variations = [phrase]
        except Exception as e:
            logger.debug(f"Failed to generate variations for '{phrase[:30]}...': {e}")
            variations = [phrase]

        self.analysis_cache[cache_key] = variations
        return variations

    def augment_with_variations(self, dialogs: list[dict]) -> list[dict]:
        """Pre-generate variations for key phrases"""
        logger.info("Generating phrase variations...")

        # Sample unique phrases to generate variations
        client_phrases = list(set(d["dialog"][0]["text"] for d in dialogs[:1000]))
        agent_phrases = list(set(d["dialog"][1]["text"] for d in dialogs[:1000]))

        # Generate variations (limited to avoid too many LLM calls)
        for phrase in client_phrases[:100]:
            self.generate_variations(phrase, "customer complaint", count=3)

        for phrase in agent_phrases[:100]:
            self.generate_variations(phrase, "support response", count=3)

        logger.success(f"Generated {len(self.analysis_cache)} variation sets")
        return dialogs

    def create_problematic_version(self, dialog: dict) -> dict:
        """Create a problematic version of a dialog"""
        new_dialog = json.loads(json.dumps(dialog))  # deep copy

        mistake = random.choice(AGENT_MISTAKES)

        # Modify agent response based on mistake type
        agent_text = new_dialog["dialog"][1]["text"]

        if mistake == "rude_tone":
            prefixes = ["Look, ", "As I said, ", "I already told you, ", "Obviously, "]
            agent_text = random.choice(prefixes) + agent_text
        elif mistake == "no_resolution":
            agent_text = "I understand your concern. Please try again later or contact us through another channel."
        elif mistake == "unnecessary_escalation":
            agent_text = "I'll need to transfer you to another department for this. Please hold."
        elif mistake == "ignored_question":
            agent_text = "Thank you for contacting us. Is there anything else I can help you with?"

        new_dialog["dialog"][1]["text"] = agent_text
        new_dialog["dialog"][0]["text"] = self._get_phrase_variation(new_dialog["dialog"][0]["text"], "customer complaint")
        new_dialog["metadata"] = {
            "intent": dialog.get("base_intent", dialog.get("metadata", {}).get("intent", "other")),
            "satisfaction": "unsatisfied",
            "quality_score": random.randint(1, 2),
            "agent_mistakes": [mistake],
            "hidden_dissatisfaction": False,
        }

        return new_dialog

    def create_hidden_dissatisfaction_version(self, dialog: dict) -> dict:
        """Create version with hidden dissatisfaction"""
        new_dialog = json.loads(json.dumps(dialog))

        # Use variations
        new_dialog["dialog"][0]["text"] = self._get_phrase_variation(new_dialog["dialog"][0]["text"], "customer complaint")
        new_dialog["dialog"][1]["text"] = self._get_phrase_variation(new_dialog["dialog"][1]["text"], "support response")

        # Add a closing message that seems positive but isn't
        closing = random.choice(HIDDEN_DISSATISFACTION_CLOSINGS)
        new_dialog["dialog"].append({"role": "client", "text": closing})

        new_dialog["metadata"] = {
            "intent": dialog.get("base_intent", dialog.get("metadata", {}).get("intent", "other")),
            "satisfaction": "unsatisfied",  # Real satisfaction
            "quality_score": random.randint(2, 3),
            "agent_mistakes": random.choice([[], ["no_resolution"]]),
            "hidden_dissatisfaction": True,
        }

        return new_dialog

    def _get_phrase_variation(self, phrase: str, variation_type: str) -> str:
        """Helper to get a cached variation or return the original phrase"""
        cache_key = f"{variation_type}:{hash(phrase)}"
        if cache_key in self.analysis_cache and self.analysis_cache[cache_key]:
            return random.choice(self.analysis_cache[cache_key])
        return phrase

    def expand_dataset(self, dialogs: list[dict], target_count: int = 100_000) -> list[dict]:
        """Expand dataset through combinatorial augmentation"""
        logger.info(f"Expanding from {len(dialogs)} to {target_count} dialogs...")

        expanded = []

        # Distribution targets
        successful_target = int(target_count * 0.50)  # 50%
        problematic_target = int(target_count * 0.30)  # 30%
        conflict_target = int(target_count * 0.15)  # 15%
        hidden_target = int(target_count * 0.05)  # 5%

        # Categorize existing dialogs
        successful_base = [d for d in dialogs if d.get("metadata", {}).get("satisfaction") == "satisfied"]
        other_base = [d for d in dialogs if d.get("metadata", {}).get("satisfaction") != "satisfied"]

        if not successful_base:
            successful_base = dialogs
        if not other_base:
            other_base = dialogs

        logger.debug(f"Base successful: {len(successful_base)}, other: {len(other_base)}")

        # 1. Successful cases
        logger.info("Generating successful cases...")
        for i in range(successful_target):
            base = random.choice(successful_base)
            new_dialog = json.loads(json.dumps(base))
            new_dialog["id"] = hashlib.md5(f"success:{SEED}:{i}".encode()).hexdigest()[:12]

            # Use variations
            new_dialog["dialog"][0]["text"] = self._get_phrase_variation(base["dialog"][0]["text"], "customer complaint")
            new_dialog["dialog"][1]["text"] = self._get_phrase_variation(base["dialog"][1]["text"], "support response")

            if "metadata" not in new_dialog:
                new_dialog["metadata"] = {}
            new_dialog["metadata"].update({
                "intent": base.get("base_intent", "other"),
                "satisfaction": "satisfied",
                "quality_score": random.randint(4, 5),
                "agent_mistakes": [],
                "hidden_dissatisfaction": False,
            })
            expanded.append(new_dialog)

        # 2. Problematic cases
        logger.info("Generating problematic cases...")
        for i in range(problematic_target):
            base = random.choice(dialogs)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"problem:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = random.choice(["neutral", "unsatisfied"])
            new_dialog["metadata"]["quality_score"] = random.randint(2, 3)
            expanded.append(new_dialog)

        # 3. Conflict cases (severe issues)
        logger.info("Generating conflict cases...")
        for i in range(conflict_target):
            base = random.choice(dialogs)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"conflict:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = "unsatisfied"
            new_dialog["metadata"]["quality_score"] = random.randint(1, 2)
            new_dialog["metadata"]["agent_mistakes"] = random.sample(AGENT_MISTAKES, k=random.randint(1, 2))
            expanded.append(new_dialog)

        # 4. Hidden dissatisfaction
        logger.info("Generating hidden dissatisfaction cases...")
        for i in range(hidden_target):
            base = random.choice(successful_base)
            new_dialog = self.create_hidden_dissatisfaction_version(base)
            new_dialog["id"] = hashlib.md5(f"hidden:{SEED}:{i}".encode()).hexdigest()[:12]
            expanded.append(new_dialog)

        random.shuffle(expanded)
        logger.success(f"Expansion complete. Total dialogs: {len(expanded)}")

        return expanded
