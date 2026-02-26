"""
Prompt templates for the dataset generator / augmenter.

All prompts used by DatasetAugmenter are defined here.
Dynamic values are injected via str.format() or f-string interpolation at call sites.
"""

import json

from src.config.constants import AGENT_MISTAKES

# Quick batch analysis (generator-side)

QUICK_BATCH_ANALYSIS_HEADER = """\
Analyze these customer support dialogs. For each, provide:
- satisfaction: "satisfied", "neutral", or "unsatisfied" \
(based on if problem seems resolved)
- quality_score: 1-5 (agent's helpfulness and professionalism)
- agent_mistakes: list from {agent_mistakes} or empty []

Return a JSON object with key "results" containing an array of objects: \
{{"results": [{{"idx": 0, "satisfaction": "...", "quality_score": N, "agent_mistakes": [...]}}, ...]}}

Dialogs:
"""

QUICK_BATCH_ANALYSIS_FOOTER = (
    '\nReturn ONLY a JSON object with "results" key containing the array. Include ALL dialogs.'
)


def build_quick_batch_prompt(batch: list[dict]) -> str:
    """Build a quick-analysis prompt for a batch of dialogs."""
    header = QUICK_BATCH_ANALYSIS_HEADER.format(
        agent_mistakes=json.dumps(AGENT_MISTAKES),
    )

    dialogs_text = ""
    for j, d in enumerate(batch):
        client_msg = d["dialog"][0]["text"]
        agent_msg = d["dialog"][1]["text"]
        dialogs_text += f"\n[{j}] Client: {client_msg[:200]}\nAgent: {agent_msg[:200]}\n"

    return header + dialogs_text + QUICK_BATCH_ANALYSIS_FOOTER


# Phrase variation

PHRASE_VARIATION = """\
Generate {count} variations of this {variation_type}.
Keep the same meaning but vary the wording and tone (polite/neutral/frustrated).
Original: "{phrase}"
Return JSON array of strings only."""


def build_variation_prompt(
    phrase: str,
    variation_type: str,
    count: int,
) -> str:
    """Build prompt for generating phrase variations."""
    return PHRASE_VARIATION.format(
        count=count,
        variation_type=variation_type,
        phrase=phrase,
    )


# Dialog extension

FOLLOWUP_SCENARIOS = [
    "The client asks a follow-up question to clarify something from the agent's response.",
    "The client doesn't fully understand and asks the agent to explain more simply.",
    "The client provides additional details about their problem.",
    "The client confirms they understood and asks about a related concern.",
    "The client is not satisfied with the answer and pushes for a better solution.",
    "The client thanks the agent but has one more question.",
]

STYLE_INSTRUCTIONS = {
    "short": ("Keep ALL messages SHORT (1-2 sentences max). Client asks brief questions, agent gives concise answers."),
    "normal": ("Use MEDIUM length messages (2-4 sentences). Natural conversational style."),
    "verbose": (
        "Client writes DETAILED messages explaining their situation "
        "at length (4-6 sentences). Agent gives thorough, "
        "comprehensive responses (5-8 sentences)."
    ),
    "mixed": (
        "VARY the length: some messages short (1 sentence), "
        "some medium (2-3 sentences), some longer (4-5 sentences). "
        "Mix it naturally."
    ),
}

EXTEND_DIALOG = """\
Continue this customer support conversation for exactly \
{extra_turns} more exchange(s).
Each exchange = 1 client message + 1 agent response.

Scenario: {scenario}
Style: {style_instruction}

Current conversation:
{conversation_text}

Return a JSON array of exactly {total_messages} message objects.
Each object has "role" ("client" or "agent") and "text".
Messages MUST alternate: client, agent, client, agent...
Return ONLY the JSON array."""


def build_extend_dialog_prompt(
    conversation_text: str,
    extra_turns: int,
    scenario: str,
    style: str,
) -> str:
    """Build prompt for extending a dialog with follow-up exchanges."""
    return EXTEND_DIALOG.format(
        extra_turns=extra_turns,
        scenario=scenario,
        style_instruction=STYLE_INSTRUCTIONS[style],
        conversation_text=conversation_text,
        total_messages=extra_turns * 2,
    )


# Problematic version

PROBLEMATIC_VERSION = """\
Rewrite this support agent response to contain this specific mistake: \
{mistake}.

Mistake definitions:
- ignored_question: Agent completely ignores the client's actual question \
and talks about something unrelated
- incorrect_info: Agent provides wrong or misleading information
- rude_tone: Agent is dismissive, condescending, or impatient
- no_resolution: Agent acknowledges the problem but doesn't provide \
a solution or useful next steps
- unnecessary_escalation: Agent immediately transfers/escalates instead \
of trying to help

Client said: {client_text}
Original agent response: {agent_text}

Rewrite ONLY the agent response (1-3 sentences) to clearly exhibit \
the '{mistake}' error.
Return JSON object: {{"agent_response": "...the rewritten response..."}}"""


def build_problematic_prompt(
    mistake: str,
    client_text: str,
    agent_text: str,
) -> str:
    """Build prompt for creating a problematic agent response."""
    return PROBLEMATIC_VERSION.format(
        mistake=mistake,
        client_text=client_text,
        agent_text=agent_text,
    )
