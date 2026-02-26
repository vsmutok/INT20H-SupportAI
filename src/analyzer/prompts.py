"""
Prompt templates for the dataset analyzer.

All prompts used by DatasetAnalyzer are defined here.
Dynamic values are injected via str.format() or f-string interpolation at call sites.
"""

import json

from src.config.constants import AGENT_MISTAKES, VALID_INTENTS

# Single dialog analysis

SINGLE_DIALOG_ANALYSIS = """You are an expert customer support quality analyst. Analyze this conversation carefully.

CONVERSATION:
{dialog_text}

Return a JSON object with ALL of these fields:

1. "intent": The customer's primary intent. Choose from: {valid_intents}

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

4. "agent_mistakes": List from {agent_mistakes} or empty []
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


def build_single_dialog_prompt(dialog_text: str) -> str:
    """Build prompt for analyzing a single dialog."""
    return SINGLE_DIALOG_ANALYSIS.format(
        dialog_text=dialog_text,
        valid_intents=json.dumps(VALID_INTENTS),
        agent_mistakes=json.dumps(AGENT_MISTAKES),
    )


# Batch dialog analysis

BATCH_DIALOG_HEADER = """You are an expert customer support quality analyst. \
Analyze each conversation below.

For each conversation determine ALL of these:
- "intent": one of {valid_intents}
- "satisfaction": "satisfied", "neutral", or "unsatisfied"
- "quality_score": 1-5 (5=excellent, 1=very poor)
- "agent_mistakes": list from {agent_mistakes} or []
- "hidden_dissatisfaction": true if customer seems polite but problem NOT resolved
- "tone_agent": "professional", "casual", or "rude"
- "tone_client": "calm", "frustrated", or "angry"
- "resolution": "resolved", "partially_resolved", or "unresolved"
- "resolution_summary": one sentence about resolution status

IMPORTANT for hidden_dissatisfaction:
If customer says "okay thanks", "I understand", "fine" but agent did NOT \
actually solve the problem, set true.

"""

BATCH_DIALOG_FOOTER = (
    "\nReturn a JSON array of exactly {batch_size} objects."
    ' Each must have "idx" (0-based index) plus all fields above.'
    "\nReturn ONLY the JSON array."
)


def build_batch_dialog_prompt(batch: list[dict]) -> str:
    """Build prompt for analyzing a batch of dialogs."""
    header = BATCH_DIALOG_HEADER.format(
        valid_intents=json.dumps(VALID_INTENTS),
        agent_mistakes=json.dumps(AGENT_MISTAKES),
    )

    conversations = ""
    for j, dialog_data in enumerate(batch):
        dialog_msgs = dialog_data.get("dialog", [])
        conversations += f"\n--- CONVERSATION {j} ---\n"
        for msg in dialog_msgs:
            conversations += f"{msg['role'].capitalize()}: {msg['text'][:300]}\n"

    footer = BATCH_DIALOG_FOOTER.format(batch_size=len(batch))

    return header + conversations + footer
