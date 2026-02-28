"""
Two-tier prompt templates for the dataset analyzer (v6).

All v1-v5 learnings combined + disambiguation hints:
- Intent: disambiguation rules for top confusion pairs (general, universal)
- Satisfaction: scenario-based hints, "neutral" stays rare
- Quality: fix score=1 overprediction, scenario hints
- Mistakes: negative examples for incorrect_info, clearer definitions
- HD: keep v4/v5 strict approach (best ever)
"""

import json

from src.config.constants import AGENT_MISTAKES, VALID_INTENTS

SYSTEM_PROMPT = (
    "You are a customer support quality analyst. "
    "You classify conversations step by step. Return ONLY valid JSON."
)

INTENT_DESCRIPTIONS = {
    "payment_issue": (
        "Problems with payments, payment methods, payment errors, "
        "or payment failures"
    ),
    "technical_error": (
        "Placing orders, purchasing items, or registration problems"
    ),
    "account_access": (
        "Creating, editing, deleting, or switching accounts; "
        "password recovery; updating personal information"
    ),
    "tariff_question": (
        "Questions about company services and information: "
        "newsletter subscribe/unsubscribe, checking invoices or bills, "
        "shipping and delivery options, delivery addresses, "
        "customer support hours, contacting support, "
        "service offerings, plans, pricing, cancellation fees"
    ),
    "refund_request": (
        "Requesting refunds, checking refund status, "
        "refund policies, or canceling orders for a refund"
    ),
    "other": (
        "Anything that does not fit above: tracking orders, "
        "checking order status or ETA, filing complaints, "
        "general inquiries, feedback, reviews"
    ),
}

INTENT_EXAMPLES = {
    "tariff_question": [
        "checking invoices or bills",
        "shipping/delivery options or methods",
        "newsletter subscribe/unsubscribe",
        "customer support hours",
        "changing delivery address",
    ],
    "technical_error": [
        "placing or purchasing an order",
        "registration problems",
    ],
    "payment_issue": [
        "payment failure or error",
        "available payment methods",
    ],
    "account_access": [
        "create/edit/delete account",
        "password recovery or login help",
        "switching account types",
    ],
    "refund_request": [
        "requesting a refund",
        "checking refund status",
        "refund policy questions",
    ],
    "other": [
        "tracking order status or ETA",
        "filing a complaint or claim",
        "general inquiry or feedback",
    ],
}

# Disambiguation rules for commonly confused intent pairs
INTENT_DISAMBIGUATION = """
IMPORTANT — when in doubt between similar intents, follow these rules:
- Delivery/shipping OPTIONS or METHODS → "tariff_question" (service info)
- Tracking a specific order or checking ETA → "other" (not tariff_question)
- Placing or creating a NEW order → "technical_error" (not other)
- Changing delivery ADDRESS → "tariff_question" (service setting)
- Editing ACCOUNT profile or personal details → "account_access"
- Asking about subscription PLANS or PRICING → "tariff_question"
- Payment FAILED or payment METHOD question → "payment_issue"
- Wants money BACK or order CANCELLATION for refund → "refund_request"
- Complaint about service quality → "other"
- Asking about invoices, bills, fees → "tariff_question"
"""

# Satisfaction scenario hints
SATISFACTION_HINTS = """
Hints for satisfaction:
- Agent gives steps like "go to Settings > Account" → "satisfied"
- Agent says "I understand, let me help" and provides info → "satisfied"
- Agent gives a generic but on-topic answer → "satisfied"
- Agent says "I'll transfer you to a specialist" without solving anything → "unsatisfied"
- Client says "this doesn't help" or "I already tried that" → "unsatisfied"
- Agent is rude or dismissive → "unsatisfied"
- Conversation ends abruptly mid-discussion → "neutral"
"""

MAX_AGENT_CHARS = 300

# ---------------------------------------------------------------------------
# Tier 1: intent + satisfaction (batch_size=15)
# ---------------------------------------------------------------------------

TIER1_PROMPT = """\
Classify each conversation below.

INTENT — classify the client's primary intent:
{intent_descriptions}

Examples of what belongs to each intent:
{intent_examples}

{intent_disambiguation}

SATISFACTION — judge by outcome (most conversations are "satisfied" or "unsatisfied", rarely "neutral"):
- "satisfied": agent responded to the question and tried to help. Even if the answer is generic or not perfect, if the agent addressed the topic and client did not complain, this is "satisfied".
- "unsatisfied": agent clearly failed — gave wrong info, was rude, ignored the question, OR client expressed frustration/disappointment.
- "neutral": ONLY when the conversation is truly ambiguous or cut short with no clear outcome. This should be rare (~15% of conversations).

{satisfaction_hints}

{conversations}

Return JSON: {{"results": [{{"idx": 0, "reason": "<what is the client asking?>", "intent": "...", "satisfaction": "..."}}, ...]}}
Return ONLY JSON. Include ALL {batch_size} conversations."""

# ---------------------------------------------------------------------------
# Tier 2: quality_score + agent_mistakes + hidden_dissatisfaction (batch_size=10)
# ---------------------------------------------------------------------------

TIER2_PROMPT = """\
Evaluate agent performance in each conversation below.

QUALITY SCORE — agent performance (use the full 1-5 range):
- 5: Excellent. Agent gives clear, specific steps that fully resolve the issue. Example: provides exact navigation path, specific policy details, or concrete solution. (~25%)
- 4: Good. Agent provides helpful guidance and mostly addresses the issue. Example: gives relevant advice but misses some details. (~25%)
- 3: Adequate. Agent acknowledges the problem and gives a reasonable but generic response. Example: "I understand your concern, please try X". (~15%)
- 2: Poor. Agent's response is unhelpful, misses the main question, or gives vague non-answers. Example: "Please contact support" without actually helping. (~25%)
- 1: Terrible. Agent is actively rude, gives CLEARLY wrong information, or completely ignores the customer. This is rare — a generic unhelpful answer is score 2, not 1. (~10%)

AGENT MISTAKES — list any that clearly occurred, or []. Only include if there is clear evidence.
- "ignored_question": agent completely ignored what the client asked about and talked about something else
- "incorrect_info": agent stated something FACTUALLY WRONG (e.g., wrong price, wrong policy, wrong procedure steps). NOT incorrect_info: generic answers, template responses, incomplete info, or simply not knowing the answer
- "rude_tone": agent was impolite, dismissive, sarcastic, or unprofessional in tone
- "no_resolution": agent acknowledged the problem but did not offer any solution, steps, or next actions
- "unnecessary_escalation": agent escalated to supervisor/manager when they could have handled the issue themselves

HIDDEN DISSATISFACTION — this is EXTREMELY RARE (expect less than 5% of conversations).
Set true ONLY if the client says something like "fine, I'll figure it out myself" or "okay whatever, thanks anyway" while their problem is clearly NOT solved.
In ALL other cases, set false. If unsure, set false.

{conversations}

Return JSON: {{"results": [{{"idx": 0, "reason": "<was the problem solved? how well?>", "quality_score": N, "agent_mistakes": [...], "hidden_dissatisfaction": false}}, ...]}}
Return ONLY JSON. Include ALL {batch_size} conversations."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_intent_descriptions() -> str:
    lines = []
    for intent, desc in INTENT_DESCRIPTIONS.items():
        lines.append(f'- "{intent}": {desc}')
    return "\n".join(lines)


def _format_intent_examples() -> str:
    lines = []
    for intent, examples in INTENT_EXAMPLES.items():
        lines.append(f'  "{intent}": {", ".join(examples)}')
    return "\n".join(lines)


def _format_conversations(batch: list[dict]) -> str:
    parts = []
    for j, dialog_data in enumerate(batch):
        dialog_msgs = dialog_data.get("dialog", [])
        parts.append(f"[{j}]")
        for msg in dialog_msgs:
            role = "C" if msg["role"] == "client" else "A"
            text = msg["text"]
            if role == "A" and len(text) > MAX_AGENT_CHARS:
                text = text[:MAX_AGENT_CHARS] + "..."
            parts.append(f"{role}: {text}")
        parts.append("")
    return "\n".join(parts)


def build_tier1_prompt(batch: list[dict]) -> str:
    return TIER1_PROMPT.format(
        intent_descriptions=_format_intent_descriptions(),
        intent_examples=_format_intent_examples(),
        intent_disambiguation=INTENT_DISAMBIGUATION.strip(),
        satisfaction_hints=SATISFACTION_HINTS.strip(),
        conversations=_format_conversations(batch),
        batch_size=len(batch),
    )


def build_tier2_prompt(batch: list[dict]) -> str:
    return TIER2_PROMPT.format(
        agent_mistakes=json.dumps(AGENT_MISTAKES),
        conversations=_format_conversations(batch),
        batch_size=len(batch),
    )


# ---------------------------------------------------------------------------
# Single dialog fallback
# ---------------------------------------------------------------------------

SINGLE_DIALOG_PROMPT = """\
Analyze this customer support conversation step by step.

INTENT categories:
{intent_descriptions}

Examples:
{intent_examples}

When in doubt between similar intents:
- Delivery/shipping options → "tariff_question"; tracking a specific order → "other"
- Placing a new order → "technical_error"; complaint about service → "other"
- Account profile/settings → "account_access"; invoices/bills/fees → "tariff_question"
- Payment failed → "payment_issue"; wants money back → "refund_request"

SATISFACTION: "satisfied" if agent addressed the topic and client didn't complain, "unsatisfied" if agent failed or client frustrated, "neutral" ONLY if truly ambiguous (rare).

QUALITY SCORE: 5=excellent specific resolution (~25%), 4=good helpful guidance (~25%), 3=adequate generic response (~15%), 2=poor unhelpful (~25%), 1=terrible rude/wrong (~10%). Use the full range.

AGENT MISTAKES: list only if clearly evident, or [].
- "ignored_question": completely ignored the question
- "incorrect_info": stated something FACTUALLY WRONG (not just vague or generic)
- "rude_tone": impolite or dismissive
- "no_resolution": offered no solution or next steps
- "unnecessary_escalation": escalated when they could handle it

HIDDEN DISSATISFACTION: EXTREMELY RARE. True only if client explicitly gives up ("fine, I'll figure it out"). Otherwise false.

CONVERSATION:
{dialog_text}

Return JSON:
{{"reason": "<what is the client asking? was it resolved?>", "intent": "...", "satisfaction": "...", "quality_score": N, "agent_mistakes": [], "hidden_dissatisfaction": false}}
Return ONLY JSON."""


def build_single_dialog_prompt(dialog_text: str) -> str:
    return SINGLE_DIALOG_PROMPT.format(
        intent_descriptions=_format_intent_descriptions(),
        intent_examples=_format_intent_examples(),
        dialog_text=dialog_text,
        valid_intents=json.dumps(VALID_INTENTS),
        agent_mistakes=json.dumps(AGENT_MISTAKES),
    )
