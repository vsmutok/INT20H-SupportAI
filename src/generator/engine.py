import json
import re
import random
import hashlib
from loguru import logger
from datasets import load_dataset
from ollama import Client

from src.config.constants import (
    SEED,
    INTENT_MAP,
    AGENT_MISTAKES,
    HIDDEN_DISSATISFACTION_CLOSINGS,
    TEMPLATE_REPLACEMENTS,
)


class DatasetAugmenter:
    def __init__(self, ollama_model: str = "llama3.1:8b"):
        self.client = Client()
        self.model = ollama_model
        self.analysis_cache: dict[str, dict] = {}
        self.rng = random.Random(SEED)
        logger.info(f"Initialized DatasetAugmenter with model: {self.model}")

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
        logger.info("Loading Bitext dataset from HuggingFace...")
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

        dialogs = []
        for sample in ds["train"]:
            intent = INTENT_MAP.get(sample.get("intent", ""), "other")

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
        logger.success(f"Loaded and shuffled {len(dialogs)} base dialogs")
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

    def analyze_dialog_batch(self, dialogs: list[dict], batch_size: int = 10) -> list[dict]:
        """Analyze dialogs in batches via LLM to add quality metrics"""
        analyzed = []
        logger.info(f"Analyzing {len(dialogs)} dialogs in batches of {batch_size}...")

        for i in range(0, len(dialogs), batch_size):
            batch = dialogs[i : i + batch_size]

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
                    options={"temperature": 0.3, "seed": SEED},
                )

                results = self._parse_json_response(response["response"])

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
                        "satisfaction": self.rng.choice(["satisfied", "satisfied", "neutral", "unsatisfied"]),
                        "quality_score": self.rng.randint(3, 5),
                        "agent_mistakes": [],
                        "hidden_dissatisfaction": False,
                    }
                    analyzed.append(d)

            if (i + batch_size) % 500 == 0:
                logger.info(f"  Analyzed {min(i + batch_size, len(dialogs))}/{len(dialogs)}")

        logger.success(f"Analysis complete. Total analyzed: {len(analyzed)}")
        return analyzed

    def generate_variations(self, phrase: str, variation_type: str, count: int = 5) -> list[str]:
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
            logger.debug(f"Failed to generate variations for '{phrase[:30]}...': {e}")
            variations = [phrase]

        self.analysis_cache[cache_key] = variations
        return variations

    def augment_with_variations(self, dialogs: list[dict]) -> list[dict]:
        """Pre-generate variations for key phrases"""
        logger.info("Generating phrase variations...")

        client_phrases = list(set(d["dialog"][0]["text"] for d in dialogs[:1000]))
        agent_phrases = list(set(d["dialog"][1]["text"] for d in dialogs[:1000]))

        for phrase in client_phrases[:100]:
            self.generate_variations(phrase, "customer complaint", count=3)

        for phrase in agent_phrases[:100]:
            self.generate_variations(phrase, "support response", count=3)

        logger.success(f"Generated {len(self.analysis_cache)} variation sets")
        return dialogs

    def _extend_dialog_turns(self, dialog: dict, extra_turns: int = 1, style: str = "normal") -> dict:
        """Use LLM to extend a dialog with realistic follow-up exchanges.

        Args:
            dialog: The dialog to extend
            extra_turns: Number of extra exchanges (1 exchange = client + agent)
            style: Message style - "short", "normal", "verbose", or "mixed"
        """
        current_messages = dialog["dialog"]
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['text'][:250]}" for msg in current_messages
        )

        style_instructions = {
            "short": "Keep ALL messages SHORT (1-2 sentences max). Client asks brief questions, agent gives concise answers.",
            "normal": "Use MEDIUM length messages (2-4 sentences). Natural conversational style.",
            "verbose": "Client writes DETAILED messages explaining their situation at length (4-6 sentences). Agent gives thorough, comprehensive responses (5-8 sentences).",
            "mixed": "VARY the length: some messages short (1 sentence), some medium (2-3 sentences), some longer (4-5 sentences). Mix it naturally.",
        }

        followup_scenarios = [
            "The client asks a follow-up question to clarify something from the agent's response.",
            "The client doesn't fully understand and asks the agent to explain more simply.",
            "The client provides additional details about their problem.",
            "The client confirms they understood and asks about a related concern.",
            "The client is not satisfied with the answer and pushes for a better solution.",
            "The client thanks the agent but has one more question.",
        ]
        scenario = self.rng.choice(followup_scenarios)

        prompt = f"""Continue this customer support conversation for exactly {extra_turns} more exchange(s).
Each exchange = 1 client message + 1 agent response.

Scenario: {scenario}
Style: {style_instructions[style]}

Current conversation:
{conversation_text}

Return a JSON array of exactly {extra_turns * 2} message objects.
Each object has "role" ("client" or "agent") and "text".
Messages MUST alternate: client, agent, client, agent...
Return ONLY the JSON array."""

        added = 0
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.7, "seed": SEED, "num_predict": 768},
            )
            turns = self._parse_json_response(response["response"])
            # Handle both dict (single message) and list responses
            if isinstance(turns, dict):
                turns = [turns]
            if isinstance(turns, list):
                for turn in turns:
                    if isinstance(turn, dict) and "role" in turn and "text" in turn:
                        if turn["role"] in ("client", "agent"):
                            # Enforce role alternation: skip if same role as last message
                            last_role = dialog["dialog"][-1]["role"] if dialog["dialog"] else None
                            if turn["role"] != last_role:
                                dialog["dialog"].append({"role": turn["role"], "text": turn["text"]})
                                added += 1
        except Exception as e:
            logger.debug(f"Failed to extend dialog via LLM: {e}")

        # Fallback: if LLM didn't produce enough messages, fill with templates
        # First ensure the last message is from the expected role
        needed = extra_turns * 2 - added
        if needed > 0:
            last_role = dialog["dialog"][-1]["role"] if dialog["dialog"] else "agent"
            for _ in range(needed):
                if last_role == "agent":
                    # Next should be client
                    msg = self.rng.choice([
                        "Could you explain that in simpler terms?",
                        "I see, but what if that doesn't work?",
                        "Thanks, and one more thing — how long will this take?",
                        "OK, but I already tried that. What else can I do?",
                        "Got it. Is there a faster way to do this?",
                        "Sorry, I'm a bit confused. Can you walk me through it step by step?",
                        "And what happens after that?",
                        "Will I receive a confirmation email?",
                    ])
                    dialog["dialog"].append({"role": "client", "text": msg})
                    last_role = "client"
                else:
                    # Next should be agent
                    msg = self.rng.choice([
                        "Of course! Let me break it down for you step by step.",
                        "I understand your concern. Let me provide an alternative solution.",
                        "Typically it takes 3-5 business days. I'll make sure to follow up with you.",
                        "I apologize for the inconvenience. Let me check another option for you right away.",
                        "Sure, there's actually a quicker method I can walk you through.",
                        "Absolutely. Once completed, you'll get a confirmation within 24 hours.",
                        "Great question. Let me explain that in more detail.",
                        "No worries at all! Here's what you need to do next.",
                    ])
                    dialog["dialog"].append({"role": "agent", "text": msg})
                    last_role = "agent"

        return dialog

    def create_problematic_version(self, dialog: dict) -> dict:
        """Create a problematic version of a dialog using LLM for realistic mistakes."""
        new_dialog = json.loads(json.dumps(dialog))
        mistake = self.rng.choice(AGENT_MISTAKES)

        client_text = new_dialog["dialog"][0]["text"][:300]
        agent_text = new_dialog["dialog"][1]["text"][:300]

        prompt = f"""Rewrite this support agent response to contain this specific mistake: {mistake}.

Mistake definitions:
- ignored_question: Agent completely ignores the client's actual question and talks about something unrelated
- incorrect_info: Agent provides wrong or misleading information
- rude_tone: Agent is dismissive, condescending, or impatient
- no_resolution: Agent acknowledges the problem but doesn't provide a solution or useful next steps
- unnecessary_escalation: Agent immediately transfers/escalates instead of trying to help

Client said: {client_text}
Original agent response: {agent_text}

Rewrite ONLY the agent response (1-3 sentences) to clearly exhibit the '{mistake}' error.
Return JSON object: {{"agent_response": "...the rewritten response..."}}"""

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

    def expand_dataset(self, dialogs: list[dict], target_count: int = 100_000) -> list[dict]:
        """Expand dataset through combinatorial augmentation with max diversity."""
        logger.info(f"Expanding from {len(dialogs)} to {target_count} dialogs...")

        expanded = []

        # Distribution targets — diverse mix
        successful_target = int(target_count * 0.45)
        problematic_target = int(target_count * 0.25)
        conflict_target = int(target_count * 0.15)
        hidden_target = int(target_count * 0.10)
        multi_turn_target = int(target_count * 0.05)

        # Categorize existing dialogs
        successful_base = [d for d in dialogs if d.get("metadata", {}).get("satisfaction") == "satisfied"]
        other_base = [d for d in dialogs if d.get("metadata", {}).get("satisfaction") != "satisfied"]

        if not successful_base:
            successful_base = dialogs
        if not other_base:
            other_base = dialogs

        # Group by intent for diversity
        by_intent: dict[str, list[dict]] = {}
        for d in dialogs:
            intent = d.get("base_intent", "other")
            by_intent.setdefault(intent, []).append(d)

        intent_keys = list(by_intent.keys())
        logger.debug(f"Base successful: {len(successful_base)}, other: {len(other_base)}, intents: {len(intent_keys)}")

        # 1. Successful cases — cycle through intents for diversity
        logger.info("Generating successful cases...")
        for i in range(successful_target):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, successful_base)
            base = self.rng.choice(pool)
            new_dialog = json.loads(json.dumps(base))
            new_dialog["id"] = hashlib.md5(f"success:{SEED}:{i}".encode()).hexdigest()[:12]

            new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
                base["dialog"][0]["text"], "customer complaint"
            )
            new_dialog["dialog"][1]["text"] = self._get_phrase_variation(
                base["dialog"][1]["text"], "support response"
            )

            target_msgs = self._pick_dialog_length()
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            if "metadata" not in new_dialog:
                new_dialog["metadata"] = {}
            new_dialog["metadata"].update({
                "intent": base.get("base_intent", "other"),
                "satisfaction": "satisfied",
                "quality_score": self.rng.randint(4, 5),
                "agent_mistakes": [],
                "hidden_dissatisfaction": False,
            })
            expanded.append(new_dialog)

        # 2. Problematic cases — use LLM for realistic mistakes
        logger.info("Generating problematic cases...")
        for i in range(problematic_target):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"problem:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = self.rng.choice(["neutral", "unsatisfied"])
            new_dialog["metadata"]["quality_score"] = self.rng.randint(2, 3)

            target_msgs = self._pick_dialog_length()
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            expanded.append(new_dialog)

        # 3. Conflict cases (severe issues)
        logger.info("Generating conflict cases...")
        for i in range(conflict_target):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = self.create_problematic_version(base)
            new_dialog["id"] = hashlib.md5(f"conflict:{SEED}:{i}".encode()).hexdigest()[:12]
            new_dialog["metadata"]["satisfaction"] = "unsatisfied"
            new_dialog["metadata"]["quality_score"] = self.rng.randint(1, 2)
            new_dialog["metadata"]["agent_mistakes"] = self.rng.sample(AGENT_MISTAKES, k=self.rng.randint(1, 3))

            target_msgs = self._pick_dialog_length()
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            expanded.append(new_dialog)

        # 4. Hidden dissatisfaction
        logger.info("Generating hidden dissatisfaction cases...")
        for i in range(hidden_target):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = [d for d in by_intent.get(intent_key, successful_base) if d.get("metadata", {}).get("satisfaction") == "satisfied"]
            if not pool:
                pool = successful_base
            base = self.rng.choice(pool)
            new_dialog = self.create_hidden_dissatisfaction_version(base)
            new_dialog["id"] = hashlib.md5(f"hidden:{SEED}:{i}".encode()).hexdigest()[:12]

            # Hidden dissatisfaction already has 3 msgs, extend some further
            target_msgs = self.rng.choice([3, 4, 5, 5, 6])
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            expanded.append(new_dialog)

        # 5. Deep multi-turn conversations (always 6-7 messages)
        logger.info("Generating deep multi-turn conversations...")
        for i in range(multi_turn_target):
            intent_key = intent_keys[i % len(intent_keys)]
            pool = by_intent.get(intent_key, dialogs)
            base = self.rng.choice(pool)
            new_dialog = json.loads(json.dumps(base))
            new_dialog["id"] = hashlib.md5(f"multiturn:{SEED}:{i}".encode()).hexdigest()[:12]

            new_dialog["dialog"][0]["text"] = self._get_phrase_variation(
                base["dialog"][0]["text"], "customer complaint"
            )
            new_dialog["dialog"][1]["text"] = self._get_phrase_variation(
                base["dialog"][1]["text"], "support response"
            )

            target_msgs = self.rng.choice([6, 7])
            new_dialog = self._extend_to_target(new_dialog, target_msgs)

            satisfaction = self.rng.choice(["satisfied", "satisfied", "neutral"])
            if "metadata" not in new_dialog:
                new_dialog["metadata"] = {}
            new_dialog["metadata"].update({
                "intent": base.get("base_intent", "other"),
                "satisfaction": satisfaction,
                "quality_score": self.rng.randint(3, 5),
                "agent_mistakes": [],
                "hidden_dissatisfaction": False,
            })
            expanded.append(new_dialog)

        self.rng.shuffle(expanded)
        logger.success(f"Expansion complete. Total dialogs: {len(expanded)}")

        return expanded
