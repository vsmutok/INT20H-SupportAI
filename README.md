# SupportAI: Dataset Generator and Analyzer

A comprehensive tool for generating and analyzing customer support dialogue datasets using Large Language Models (LLM) and combinatorial augmentation. This project evaluates support quality, identifies agent mistakes, and models complex customer interactions including hidden dissatisfaction.

## Why SupportAI?

Generating high-quality, diverse support datasets is hard. SupportAI solves this by:
- **Augmenting Real Data**: Instead of generating purely synthetic data, we start with the Bitext dataset and enrich it with LLM-generated nuances.
- **Modeling Real Support Issues**: We explicitly generate scenarios with agent mistakes and hidden customer dissatisfaction to test how well your evaluation systems perform.
- **Double-Loop Validation**: We use one LLM configuration to generate data and another (independent) pass to analyze it, allowing for accuracy benchmarking.
## Key Features

### Generator (`generate.py`)
- **Synthetic Data Generation**: Creates 300 realistic customer support dialogues across 6 intent categories:
  - `payment_issue`, `technical_error`, `account_access`, `tariff_question`, `refund_request`, `other`
- **Multi-turn Dialogs**: Varied dialog lengths (2-7 messages) with LLM-generated follow-up exchanges
- **Message Style Variation**: Short, normal, verbose, and mixed message styles for both client and agent
- **Scenario Diversity**: 45% successful, 25% problematic, 15% conflict, 10% hidden dissatisfaction, 5% deep multi-turn
- **Agent Mistakes**: LLM-generated realistic agent errors: `ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation`
- **Hidden Dissatisfaction**: Cases where the client formally accepts but the problem remains unresolved
- **Template Variable Replacement**: Realistic values for order numbers, names, amounts, etc.
- **Deterministic Results**: Fixed seeds (`SEED=42`) for both Python random and LLM generation

### Analyzer (`analyze.py`)
- **Independent LLM Analysis**: Each dialog is analyzed from scratch using the LLM (not copying generator metadata)
- **Intent Classification**: Detects the customer's primary intent from the conversation content
- **Satisfaction Assessment**: Determines real customer satisfaction: `satisfied`, `neutral`, `unsatisfied`
- **Quality Scoring**: 1-5 scale with adjustments for agent mistakes and resolution status
- **Agent Mistake Detection**: Identifies specific errors from the conversation
- **Hidden Dissatisfaction Detection**: Detects when customers appear polite but are actually unsatisfied
- **Tone Analysis**: Agent tone (`professional`/`casual`/`rude`) and client tone (`calm`/`frustrated`/`angry`)
- **Resolution Tracking**: `resolved`, `partially_resolved`, `unresolved` with text summary
- **Comparison Metrics**: Accuracy comparison between analyzer output and generator metadata

## Project Structure

```text
.
├── data/                       # Generated datasets and analysis results
│   ├── dataset.json            # Generated dialog dataset
│   ├── dataset_stats.json      # Generation statistics
│   ├── analysis.json           # Analysis results (dialog + independent analysis)
│   └── analysis_stats.json     # Aggregated analysis statistics
├── src/
│   ├── generator/              # Dataset generation logic
│   │   ├── main.py             # Generator orchestrator (target_count=300)
│   │   └── engine.py           # Core: augmentation, LLM dialog extension, mistakes
│   ├── analyzer/               # Dataset analysis logic
│   │   ├── main.py             # Analyzer orchestrator + stats computation
│   │   └── engine.py           # Core: LLM-based dialog analysis + validation
│   └── config/
│       ├── __init__.py
│       └── constants.py        # Intent mappings, mistake types, template replacements
├── generate.py                 # Entry point: dataset generation
├── analyze.py                  # Entry point: dataset analysis
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Dependencies
└── README.md
```

## Docker Support

The easiest way to run the project with all its dependencies is using Docker Compose.

### Quick Start with Docker
1. **Prerequisites**: Ensure you have Docker and Docker Compose installed.
2. **Launch**:
   ```bash
   docker-compose up -d
   ```
   This will start:
   - `ollama`: The LLM engine.
   - `ollama-pull-model`: A helper service that automatically pulls `llama3.1:8b`.

3. **Run Generation**:
   Once the model is pulled (check `ollama` logs or wait a minute), run:
   ```bash
   docker-compose run app python generate.py
   ```

4. **Run Analysis**:
   ```bash
   docker-compose run app python analyze.py
   ```

Data and logs are persisted in `./data` and `./logs` directories on your host.

---

## Installation (Local)

1. **Python**: Version 3.12 or higher.
2. **Ollama**: Install and run Ollama locally with the `llama3.1:8b` model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. **Dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync

   # Using pip
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Dataset
```bash
uv run python generate.py
```
This will:
1. Load base data from the Bitext customer support dataset (HuggingFace)
2. Replace template variables with realistic values
3. Analyze a subset with LLM for quality metrics
4. Generate phrase variations for augmentation
5. Expand to 300 dialogs with diverse scenarios, lengths (2-7 messages), and styles
6. Save to `data/dataset.json` and `data/dataset_stats.json`

### 2. Analyze Dataset
```bash
uv run python analyze.py
```
This will:
1. Load `data/dataset.json`
2. Analyze each dialog independently via LLM (batches of 5)
3. Validate and normalize all analysis fields
4. Compute adjusted quality scores (penalties for mistakes, bonuses for resolution)
5. Save to `data/analysis.json` and `data/analysis_stats.json`
6. Compare analyzer output vs generator metadata (accuracy metrics)

## Output Format

### Dataset (`data/dataset.json`)
Each dialog entry contains:
```json
{
  "id": "6f1ecba24639",
  "dialog": [
    {"role": "client", "text": "I need help with my payment..."},
    {"role": "agent", "text": "I'd be happy to help..."},
    {"role": "client", "text": "Can you explain that more simply?"},
    {"role": "agent", "text": "Of course! Here's what you need to do..."}
  ],
  "metadata": {
    "intent": "payment_issue",
    "satisfaction": "satisfied",
    "quality_score": 4,
    "agent_mistakes": [],
    "hidden_dissatisfaction": false
  }
}
```

### Analysis (`data/analysis.json`)
Each analyzed entry contains:
```json
{
  "id": "6f1ecba24639",
  "dialog": [...],
  "analysis": {
    "intent": "payment_issue",
    "satisfaction": "satisfied",
    "quality_score_raw": 4,
    "quality_score": 5,
    "agent_mistakes": [],
    "hidden_dissatisfaction": false,
    "tone_agent": "professional",
    "tone_client": "calm",
    "resolution": "resolved",
    "resolution_summary": "Agent provided clear payment instructions and resolved the issue."
  }
}
```

## Documentation
- [Generation Process](docs/GENERATION_PROCESS.md) — detailed description of dataset generation steps
- [Analysis Process](docs/ANALYSIS_PROCESS.md) — detailed description of dataset analysis steps
-
## Configuration

- **Model**: Change `ollama_model` in `src/generator/main.py` or `src/analyzer/main.py`
- **Intents**: Edit `INTENT_MAP` and `VALID_INTENTS` in `src/config/constants.py`
- **Target Size**: Adjust `target_count` in `src/generator/main.py`
- **Mistake Types**: Edit `AGENT_MISTAKES` in `src/config/constants.py`
- **Dialog Lengths**: Modify `_pick_dialog_length()` weights in `src/generator/engine.py`
