import hashlib
import json
import os
import random
import re

from datasets import load_dataset
from ollama import Client

from src.config.constants import (
    AGENT_MISTAKES,
    HIDDEN_DISSATISFACTION_CLOSINGS,
    INTENT_MAP,
    SEED,
    TEMPLATE_REPLACEMENTS,
)
from src.config.logger import logger
from src.generator.prompts import (
    FOLLOWUP_SCENARIOS,
    build_extend_dialog_prompt,
    build_problematic_prompt,
    build_quick_batch_prompt,
    build_variation_prompt,
)


class DatasetAugmenter:
    def __init__(self, ollama_model: str = "llama3.1:8b", ollama_host: str | None = None):
        host = ollama_host or os.environ.get("OLLAMA_HOST")
        self.client = Client(host=host)
        self.model = ollama_model
        self.analysis_cache: dict[str, dict] = {}
        self.rng = random.Random(SEED)
        logger.info(f"Initialized DatasetAugmenter with model: {self.model} (host: {host or 'default'})")

    def _replace_template_vars(self, text: str) -> str:
        """Replace {{Variable}} placeholders with realistic values."""

        def replacer(match: re.Match) -> str:
            var = match.group(0)
            if var in TEMPLATE_REPLACEMENTS:
                return self.rng.choice(TEMPLATE_REPLACEMENTS[var])
            inner = match.group(1).strip()
            return inner

        return re.sub(r"\{\{([^}]+)\}\}", replacer, text)

    def load_base_dataset(self) -> list[dict]:
        """Load and transform Bitext dataset"""
        logger.info("Downloading Bitext dataset from HuggingFace...")
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        raw_count = len(ds["train"])
        logger.info(f"Downloaded {raw_count} samples. Transforming to dialog format...")

        dialogs = []
        intent_counts: dict[str, int] = {}
        for sample in ds["train"]:
            intent = INTENT_MAP.get(sample.get("intent", ""), "other")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

            client_text = self._replace_template_vars(sample["instruction"])
            agent_text = self._replace_template_vars(sample["response"])

            dialog = {
                "dialog": [
                    {"role": "client", "text": client_text},
                    {"role": "agent", "text": agent_text},
                ],
                "base_intent": intent,
                "original_intent": sample.get("intent", "unknown"),
            }
            dialogs.append(dialog)

        # Shuffle to ensure diverse intent distribution from the start
        self.rng.shuffle(dialogs)
        logger.success(f"Loaded {len(dialogs)} base dialogs, shuffled for diversity")
        logger.info(f"  Intent distribution: {intent_counts}")
        return dialogs

    def _parse_json_response(self, response_text: str):
        """Helper to parse JSON from LLM response, handling potential formatting issues"""
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
            start_bracket = min(candidates) if candidates else -1

            last_sq = text.rfind("]")
            last_br = text.rfind("}")
            end_candidates = [i for i in [last_sq, last_br] if i != -1]
            end_bracket = max(end_candidates) if end_candidates else -1

            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                try:
                    return json.loads(text[start_bracket : end_bracket + 1])
                except json.JSONDecodeError:
                    pass
            raise

    def _process_batch_results(self, results: list, batch: list[dict]) -> list[dict]:
        """Process parsed LLM results and attach metadata to batch dialogs."""
        analyzed = []
        matched_indices: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            idx = result.get("idx", 0)
            if idx < len(batch) and idx not in matched_indices:
                matched_indices.add(idx)
                batch[idx]["metadata"] = {
                    "intent": batch[idx]["base_intent"],
                    "satisfaction": result.get("satisfaction", "neutral"),
                    "quality_score": result.get("quality_score", 3),
                    "agent_mistakes": result.get("agent_mistakes", []),
                    "hidden_dissatisfaction": False,
                }
                analyzed.append(batch[idx])

        # Fallback for any dialogs in the batch not covered by LLM results
        for idx, d in enumerate(batch):
            if idx not in matched_indices:
                d["metadata"] = {
                    "intent": d["base_intent"],
                    "satisfaction": self.rng.choice(["satisfied", "satisfied", "neutral", "unsatisfied"]),
                    "quality_score": self.rng.randint(3, 5),
                    "agent_mistakes": [],
                    "hidden_dissatisfaction": False,
                }
                analyzed.append(d)

        return analyzed

    def _fallback_metadata(self, batch: list[dict]) -> list[dict]:
        """Assign fallback metadata when LLM analysis fails."""
        for d in batch:
            d["metadata"] = {
                "intent": d["base_intent"],
                "satisfaction": self.rng.choice(["satisfied", "satisfied", "neutral", "unsatisfied"]),
                "quality_score": self.rng.randint(3, 5),
                "agent_mistakes": [],
                "hidden_dissatisfaction": False,
            }
        return batch

    def analyze_dialog_batch(self, dialogs: list[dict], batch_size: int = 10) -> list[dict]:
        """Analyze dialogs in batches via LLM to add quality metrics"""
        analyzed = []
        total = len(dialogs)
        num_batches = (total + batch_size - 1) // batch_size
        logger.info(f"Analyzing {total} dialogs in {num_batches} batches (batch_size={batch_size})...")

        for i in range(0, total, batch_size):
            batch = dialogs[i : i + batch_size]
            batch_num = i // batch_size + 1

            prompt = build_quick_batch_prompt(batch)

            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.3, "seed": SEED},
                )

                results = self._parse_json_response(response["response"])

                # Handle wrapped format {"results": [...]}
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                elif isinstance(results, dict):
                    results = [results]
                elif not isinstance(results, list):
                    logger.warning(f"Unexpected JSON format from LLM for batch starting at {i}: {type(results)}")
                    results = []

                analyzed.extend(self._process_batch_results(results, batch))

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Batch {batch_num}/{num_batches}: Failed to parse LLM response: {e}. "
                    f"Using fallback values for {len(batch)} dialogs."
                )
                analyzed.extend(self._fallback_metadata(batch))

            done = min(i + batch_size, total)
            pct = done / total * 100
            logger.info(
                f"  Batch {batch_num}/{num_batches} complete — "
                f"{done}/{total} dialogs ({pct:.0f}%), "
                f"analyzed so far: {len(analyzed)}"
            )

        logger.success(f"Analysis complete. Total analyzed: {len(analyzed)} / {total}")
        return analyzed

    def generate_variations(self, phrase: str, variation_type: str, count: int = 5) -> list[str]:
        """Generate phrase variations for augmentation (single phrase, used as fallback)"""
        cache_key = f"{variation_type}:{hash(phrase)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        prompt = build_variation_prompt(phrase, variation_type, count)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.7, "seed": SEED},
            )
            results = self._parse_json_response(response["response"])
            if isinstance(results, list):
                variations = [str(v) for v in results]
            elif isinstance(results, str):
                variations = [results]
            else:
                variations = [phrase]
        except Exception as e:
            logger.debug(f"Variation generation failed for '{phrase[:50]}...' (type={variation_type}): {e}")
            variations = [phrase]

        self.analysis_cache[cache_key] = variations
        return variations

    def _build_variation_batch_prompt(self, batch: list[tuple[int, str]], variation_type: str, count: int) -> str:
        """Build a single prompt requesting variations for a batch of phrases."""
        prompt = (
            f"Generate {count} short variations for each of the following {variation_type} phrases.\n"
            f"Keep the same meaning but vary wording and tone.\n"
            f"Return a JSON object where each key is the phrase index "
            f"and value is an array of {count} variation strings.\n\n"
        )
        for j, (_, phrase) in enumerate(batch):
            prompt += f'[{j}]: "{phrase[:200]}"\n'
        prompt += '\nReturn ONLY a JSON object like {"0": ["var1", ...], "1": [...], ...}'
        return prompt

    def _cache_variation_results(self, result: dict, batch: list[tuple[int, str]], variation_type: str) -> None:
        """Store parsed LLM variation results into the cache."""
        for j, (_, phrase) in enumerate(batch):
            cache_key = f"{variation_type}:{hash(phrase)}"
            variations = result.get(str(j), result.get(j))
            if isinstance(variations, list) and variations:
                self.analysis_cache[cache_key] = [str(v) for v in variations]
            else:
                self.analysis_cache[cache_key] = [phrase]

    def _cache_fallback_variations(self, batch: list[tuple[int, str]], variation_type: str) -> None:
        """Cache original phrases as fallback when variation generation fails."""
        for _, phrase in batch:
            cache_key = f"{variation_type}:{hash(phrase)}"
            self.analysis_cache[cache_key] = [phrase]

    def generate_variations_batch(
        self, phrases: list[str], variation_type: str, count: int = 3, batch_size: int = 10
    ) -> None:
        """Generate variations for multiple phrases in batched LLM calls."""
        uncached = [(i, p) for i, p in enumerate(phrases) if f"{variation_type}:{hash(p)}" not in self.analysis_cache]
        if not uncached:
            return

        total = len(uncached)
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(0, total, batch_size):
            batch = uncached[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            prompt = self._build_variation_batch_prompt(batch, variation_type, count)

            try:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.7, "seed": SEED, "num_predict": 2048},
                )
                result = self._parse_json_response(response["response"])

                if isinstance(result, dict):
                    self._cache_variation_results(result, batch, variation_type)
                else:
                    self._cache_fallback_variations(batch, variation_type)
            except Exception as e:
                logger.debug(f"Batch variation generation failed (batch {batch_num}): {e}")
                self._cache_fallback_variations(batch, variation_type)

            if batch_num % 2 == 0 or batch_num == num_batches:
                done = min(batch_idx + batch_size, total)
                logger.info(f"    {variation_type}: {done}/{total} phrases ({batch_num}/{num_batches} batches)")

    def augment_with_variations(self, dialogs: list[dict]) -> list[dict]:
        """Pre-generate variations for key phrases using batched LLM calls."""
        logger.info("Generating phrase variations for augmentation...")

        client_phrases = list(set(d["dialog"][0]["text"] for d in dialogs[:1000]))
        agent_phrases = list(set(d["dialog"][1]["text"] for d in dialogs[:1000]))
        logger.info(f"  Unique client phrases: {len(client_phrases)}, unique agent phrases: {len(agent_phrases)}")

        client_limit = min(100, len(client_phrases))
        agent_limit = min(100, len(agent_phrases))

        logger.info(f"  Generating variations for {client_limit} client phrases (batched)...")
        self.generate_variations_batch(client_phrases[:client_limit], "customer complaint", count=3, batch_size=10)

        logger.info(f"  Generating variations for {agent_limit} agent phrases (batched)...")
        self.generate_variations_batch(agent_phrases[:agent_limit], "support response", count=3, batch_size=10)

        logger.success(f"Generated {len(self.analysis_cache)} variation sets")
        return dialogs

    def _parse_llm_turns(self, turns, dialog: dict) -> int:
        """Parse LLM-generated turns and append valid ones to dialog. Returns count added."""
        if isinstance(turns, dict):
            turns = [turns]
        if not isinstance(turns, list):
            return 0
        added = 0
        for turn in turns:
            if isinstance(turn, dict) and "role" in turn and "text" in turn and turn["role"] in ("client", "agent"):
                last_role = dialog["dialog"][-1]["role"] if dialog["dialog"] else None
                if turn["role"] != last_role:
                    dialog["dialog"].append({"role": turn["role"], "text": turn["text"]})
                    added += 1
        return added

    def _fill_fallback_turns(self, dialog: dict, needed: int) -> None:
        """Fill remaining turns with template messages."""
        last_role = dialog["dialog"][-1]["role"] if dialog["dialog"] else "agent"
        client_templates = [
            "Could you explain that in simpler terms?",
            "I see, but what if that doesn't work?",
            "Thanks, and one more thing — how long will this take?",
            "OK, but I already tried that. What else can I do?",
            "Got it. Is there a faster way to do this?",
            "Sorry, I'm a bit confused. Can you walk me through it step by step?",
            "And what happens after that?",
            "Will I receive a confirmation email?",
        ]
        agent_templates = [
            "Of course! Let me break it down for you step by step.",
            "I understand your concern. Let me provide an alternative solution.",
            "Typically it takes 3-5 business days. I'll make sure to follow up with you.",
            "I apologize for the inconvenience. Let me check another option for you right away.",
            "Sure, there's actually a quicker method I can walk you through.",
            "Absolutely. Once completed, you'll get a confirmation within 24 hours.",
            "Great question. Let me explain that in more detail.",
            "No worries at all! Here's what you need to do next.",
        ]
        for _ in range(needed):
            if last_role == "agent":
                msg = self.rng.choice(client_templates)
                dialog["dialog"].append({"role": "client", "text": msg})
                last_role = "client"
            else:
                msg = self.rng.choice(agent_templates)
                dialog["dialog"].append({"role": "agent", "text": msg})
                last_role = "agent"

    def _extend_dialog_turns(self, dialog: dict, extra_turns: int = 1, style: str = "normal") -> dict:
        """Use LLM to extend a dialog with realistic follow-up exchanges.

        Args:
            dialog: The dialog to extend
            extra_turns: Number of extra exchanges (1 exchange = client + agent)
            style: Message style - "short", "normal", "verbose", or "mixed"
        """
        current_messages = dialog["dialog"]
        conversation_text = "\n".join(f"{msg['role'].capitalize()}: {msg['text'][:250]}" for msg in current_messages)

        scenario = self.rng.choice(FOLLOWUP_SCENARIOS)
        prompt = build_extend_dialog_prompt(
            conversation_text=conversation_text,
            extra_turns=extra_turns,
            scenario=scenario,
            style=style,
        )

        added = 0
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.7, "seed": SEED, "num_predict": 768},
            )
            turns = self._parse_json_response(response["response"])
            added = self._parse_llm_turns(turns, dialog)
        except Exception as e:
            logger.debug(f"Failed to extend dialog via LLM: {e}")

        # Fallback: if LLM didn't produce enough messages, fill with templates
        needed = extra_turns * 2 - added
        if needed > 0:
            self._fill_fallback_turns(dialog, needed)

        return dialog

    def create_problematic_version(self, dialog: dict) -> dict:
        """Create a problematic version of a dialog using LLM for realistic mistakes."""
        new_dialog = json.loads(json.dumps(dialog))
        mistake = self.rng.choice(AGENT_MISTAKES)

        client_text = new_dialog["dialog"][0]["text"][:300]
        agent_text = new_dialog["dialog"][1]["text"][:300]

        prompt = build_problematic_prompt(
            mistake=mistake,
            client_text=client_text,
            agent_text=agent_text,
        )

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.5, "seed": SEED},
            )
            result = self._parse_json_response(response["response"])
            if isinstance(result, dict) and "agent_response" in result:
                new_dialog["dialog"][1]["text"] = result["agent_response"]
        except Exception as e:
            logger.debug(f"Falling back to template for mistake '{mistake}': {e}")
            # Fallback: use simple template-based approach
            if mistake == "rude_tone":
                prefixes = ["Look, ", "As I said, ", "I already told you, ", "Obviously, "]
                new_dialog["dialog"][1]["text"] = self.rng.choice(prefixes) + agent_text
            elif mistake == "no_resolution":
                new_dialog["dialog"][1]["text"] = (
                    "I understand your concern. Please try again later or contact us through another channel."
                )
            elif mistake == "unnecessary_escalation":
                new_dialog["dialog"][1]["text"] = (
                    "I'll need to transfer you to another department for this. Please hold."
                )
            elif mistake == "ignored_question":
                new_dialog["dialog"][1]["text"] = (
                    "Thank you for contacting us. Is there anything else I can help you with?"
                )

        new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
            new_dialog["dialog"][0]["text"], "customer complaint"
        )
        new_dialog["metadata"] = {
            "intent": dialog.get("base_intent", dialog.get("metadata", {}).get("intent", "other")),
            "satisfaction": "unsatisfied",
            "quality_score": self.rng.randint(1, 2),
            "agent_mistakes": [mistake],
            "hidden_dissatisfaction": False,
        }

        return new_dialog

    def create_hidden_dissatisfaction_version(self, dialog: dict) -> dict:
        """Create version with hidden dissatisfaction"""
        new_dialog = json.loads(json.dumps(dialog))

        new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
            new_dialog["dialog"][0]["text"], "customer complaint"
        )
        new_dialog["dialog"][1]["text"] = self._get_phrase_variation(
            new_dialog["dialog"][1]["text"], "support response"
        )

        closing = self.rng.choice(HIDDEN_DISSATISFACTION_CLOSINGS)
        new_dialog["dialog"].append({"role": "client", "text": closing})

        new_dialog["metadata"] = {
            "intent": dialog.get("base_intent", dialog.get("metadata", {}).get("intent", "other")),
            "satisfaction": "unsatisfied",
            "quality_score": self.rng.randint(2, 3),
            "agent_mistakes": self.rng.choice([[], ["no_resolution"]]),
            "hidden_dissatisfaction": True,
        }

        return new_dialog

    def _get_phrase_variation(self, phrase: str, variation_type: str) -> str:
        """Helper to get a cached variation or return the original phrase"""
        cache_key = f"{variation_type}:{hash(phrase)}"
        if cache_key in self.analysis_cache and self.analysis_cache[cache_key]:
            return self.rng.choice(self.analysis_cache[cache_key])
        return phrase

    def _pick_dialog_length(self) -> int:
        """Pick a random target message count from 2 to 7 (weighted distribution)."""
        # 2 msgs: ~15%, 3: ~20%, 4: ~25%, 5: ~20%, 6: ~12%, 7: ~8%
        return self.rng.choices([2, 3, 4, 5, 6, 7], weights=[15, 20, 25, 20, 12, 8])[0]

    def _pick_style(self) -> str:
        """Pick a random message length style."""
        return self.rng.choice(["short", "normal", "normal", "verbose", "mixed"])

    def _extend_to_target(self, dialog: dict, target_msgs: int) -> dict:
        """Extend dialog to reach target message count."""
        current = len(dialog["dialog"])
        if current >= target_msgs:
            return dialog
        # Each extra turn adds 2 messages (client + agent)
        extra_turns = (target_msgs - current) // 2
        if extra_turns > 0:
            style = self._pick_style()
            dialog = self._extend_dialog_turns(dialog, extra_turns=extra_turns, style=style)
        return dialog

    def _categorize_dialogs(self, dialogs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
        """Categorize dialogs by satisfaction and intent."""
        successful_base = [d for d in dialogs if d.get("metadata", {}).get("satisfaction") == "satisfied"]
        if not successful_base:
            successful_base = dialogs

        by_intent: dict[str, list[dict]] = {}
        for d in dialogs:
            intent = d.get("base_intent", "other")
            by_intent.setdefault(intent, []).append(d)

        return successful_base, by_intent

    def _generate_successful_cases(
        self,
        count: int,
        by_intent: dict,
        intent_keys: list,
        successful_base: list,
    ) -> list[dict]:
        """Generate successful resolution cases."""
        cases = []
        for i in range(count):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, successful_base)
            base = self.rng.choice(pool)
            new_dialog = json.loads(json.dumps(base))
            new_dialog["id"] = hashlib.md5(f"success:{SEED}:{i}".encode()).hexdigest()[:12]

            new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
                base["dialog"][0]["text"], "customer complaint"
            )
            new_dialog["dialog"][1]["text"] = self._get_phrase_variation(base["dialog"][1]["text"], "support response")

            new_dialog = self._extend_to_target(new_dialog, self._pick_dialog_length())

            if "metadata" not in new_dialog:
                new_dialog["metadata"] = {}
            new_dialog["metadata"].update(
                {
                    "intent": base.get("base_intent", "other"),
                    "satisfaction": "satisfied",
                    "quality_score": self.rng.randint(4, 5),
                    "agent_mistakes": [],
                    "hidden_dissatisfaction": False,
                }
            )
            cases.append(new_dialog)
        return cases

    def _generate_problematic_cases(
        self,
        count: int,
        by_intent: dict,
        intent_keys: list,
        dialogs: list,
    ) -> list[dict]:
        """Generate problematic interaction cases."""
        cases = []
        for i in range(count):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"problem:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = self.rng.choice(["neutral", "unsatisfied"])
            new_dialog["metadata"]["quality_score"] = self.rng.randint(2, 3)
            new_dialog = self._extend_to_target(new_dialog, self._pick_dialog_length())
            cases.append(new_dialog)
        return cases

    def _generate_conflict_cases(
        self,
        count: int,
        by_intent: dict,
        intent_keys: list,
        dialogs: list,
    ) -> list[dict]:
        """Generate severe conflict cases."""
        cases = []
        for i in range(count):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"conflict:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = "unsatisfied"
            new_dialog["metadata"]["quality_score"] = self.rng.randint(1, 2)
            new_dialog["metadata"]["agent_mistakes"] = self.rng.sample(AGENT_MISTAKES, k=self.rng.randint(1, 3))
            new_dialog = self._extend_to_target(new_dialog, self._pick_dialog_length())
            cases.append(new_dialog)
        return cases

    def _generate_hidden_cases(
        self,
        count: int,
        by_intent: dict,
        intent_keys: list,
        successful_base: list,
    ) -> list[dict]:
        """Generate hidden dissatisfaction cases."""
        cases = []
        for i in range(count):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = [
                d
                for d in by_intent.get(intent_key, successful_base)
                if d.get("metadata", {}).get("satisfaction") == "satisfied"
            ]
            if not pool:
                pool = successful_base
            base = self.rng.choice(pool)
            new_dialog = self.create_hidden_dissatisfaction_version(base)
            new_dialog["id"] = hashlib.md5(f"hidden:{SEED}:{i}".encode()).hexdigest()[:12]
            target_msgs = self.rng.choice([3, 4, 5, 5, 6])
            new_dialog = self._extend_to_target(new_dialog, target_msgs)
            cases.append(new_dialog)
        return cases

    def _generate_multiturn_cases(
        self,
        count: int,
        by_intent: dict,
        intent_keys: list,
        dialogs: list,
    ) -> list[dict]:
        """Generate deep multi-turn conversation cases."""
        cases = []
        for i in range(count):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = json.loads(json.dumps(base))
            new_dialog["id"] = hashlib.md5(f"multiturn:{SEED}:{i}".encode()).hexdigest()[:12]

            new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
                base["dialog"][0]["text"], "customer complaint"
            )
            new_dialog["dialog"][1]["text"] = self._get_phrase_variation(base["dialog"][1]["text"], "support response")

            target_msgs = self.rng.choice([6, 7])
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            satisfaction = self.rng.choice(["satisfied", "satisfied", "neutral"])
            if "metadata" not in new_dialog:
                new_dialog["metadata"] = {}
            new_dialog["metadata"].update(
                {
                    "intent": base.get("base_intent", "other"),
                    "satisfaction": satisfaction,
                    "quality_score": self.rng.randint(3, 5),
                    "agent_mistakes": [],
                    "hidden_dissatisfaction": False,
                }
            )
            cases.append(new_dialog)
        return cases

    def expand_dataset(self, dialogs: list[dict], target_count: int = 100_000) -> list[dict]:
        """Expand dataset through combinatorial augmentation with max diversity."""
        logger.info(f"Expanding from {len(dialogs)} to {target_count} dialogs...")

        # Distribution targets — diverse mix
        successful_target = int(target_count * 0.45)
        problematic_target = int(target_count * 0.25)
        conflict_target = int(target_count * 0.15)
        hidden_target = int(target_count * 0.10)
        multi_turn_target = int(target_count * 0.05)

        logger.info(
            f"  Target distribution — "
            f"successful: {successful_target}, "
            f"problematic: {problematic_target}, "
            f"conflict: {conflict_target}, "
            f"hidden: {hidden_target}, "
            f"multi-turn: {multi_turn_target}"
        )

        successful_base, by_intent = self._categorize_dialogs(dialogs)
        intent_keys = list(by_intent.keys())
        logger.info(
            f"  Base pool — successful: {len(successful_base)}, intents: {len(intent_keys)} ({', '.join(intent_keys)})"
        )

        logger.info(f"  [1/5] Generating {successful_target} successful cases...")
        expanded = self._generate_successful_cases(
            successful_target,
            by_intent,
            intent_keys,
            successful_base,
        )
        logger.info(f"    Done — {len(expanded)} dialogs so far")

        logger.info(f"  [2/5] Generating {problematic_target} problematic cases...")
        problematic = self._generate_problematic_cases(
            problematic_target,
            by_intent,
            intent_keys,
            dialogs,
        )
        expanded.extend(problematic)
        logger.info(f"    Done — {len(expanded)} dialogs so far")

        logger.info(f"  [3/5] Generating {conflict_target} conflict cases...")
        conflicts = self._generate_conflict_cases(
            conflict_target,
            by_intent,
            intent_keys,
            dialogs,
        )
        expanded.extend(conflicts)
        logger.info(f"    Done — {len(expanded)} dialogs so far")

        logger.info(f"  [4/5] Generating {hidden_target} hidden dissatisfaction cases...")
        hidden = self._generate_hidden_cases(
            hidden_target,
            by_intent,
            intent_keys,
            successful_base,
        )
        expanded.extend(hidden)
        logger.info(f"    Done — {len(expanded)} dialogs so far")

        logger.info(f"  [5/5] Generating {multi_turn_target} deep multi-turn conversations...")
        multiturn = self._generate_multiturn_cases(
            multi_turn_target,
            by_intent,
            intent_keys,
            dialogs,
        )
        expanded.extend(multiturn)

        self.rng.shuffle(expanded)
        logger.success(f"Expansion complete. Total dialogs: {len(expanded)}")

        return expanded
