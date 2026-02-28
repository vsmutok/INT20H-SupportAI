# 🏆 SupportAI: Dataset Generator and Analyzer

> **Built for the [INT20H](https://best.kpi.ua/int20h-2026/) hackathon — one of the largest student hackathons in Ukraine.**

A comprehensive tool for generating and analyzing customer support dialogue datasets using Large Language Models (LLM) and combinatorial augmentation. This project evaluates support quality, identifies agent mistakes, and models complex customer interactions including hidden dissatisfaction.

## 📋 Hackathon Challenge

A company wants to automate the analysis of its customer support operations. The goal is to generate a dataset of client–agent chat dialogs and build a tool that evaluates support quality. The system must determine not only the topic of each request, but also the real level of customer satisfaction and the quality of the agent's response.

**Core requirements:**

1. **Generate a chat dataset** between a client and a support agent with diverse scenarios:
   - Payment issues, technical errors, account access, tariff questions, refund requests
   - Successful, problematic, conflict cases, and cases with agent mistakes

2. **Analyze each dialog** and determine:
   - `intent` of the request (category or `other`)
   - `satisfaction` of the client (`satisfied` / `neutral` / `unsatisfied`)
   - `quality_score` of the support agent's work (scale 1–5)
   - `agent_mistakes` — list of agent errors (`ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation`)

3. **Important details:**
   - Some dialogs must contain **hidden dissatisfaction** (the client formally thanks the agent, but the problem remains unresolved)
   - Some cases must contain tonal or logical **agent mistakes**
   - LLM usage is mandatory; results must be deterministic

> Full task description: [`AI_TestTask.pdf`](AI_TestTask.pdf)

---

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
- **Two-Tier Architecture**: Splits analysis into two specialized LLM passes for speed and accuracy:
  - **Tier 1** (batch_size=15): Intent classification + satisfaction assessment
  - **Tier 2** (batch_size=10): Quality scoring + agent mistake detection + hidden dissatisfaction
- **Chain-of-thought reasoning**: LLM explains its reasoning before classifying, improving accuracy
- **Intent Classification**: Detects the customer's primary intent with disambiguation rules for commonly confused categories
- **Satisfaction Assessment**: Determines real customer satisfaction (`satisfied`, `neutral`, `unsatisfied`) with scenario-based hints
- **Quality Scoring**: 1-5 scale based on agent helpfulness and response quality
- **Agent Mistake Detection**: Identifies specific errors (`ignored_question`, `incorrect_info`, `rude_tone`, `no_resolution`, `unnecessary_escalation`) with strict evidence-based definitions
- **Hidden Dissatisfaction Detection**: Detects when customers appear polite but are actually unsatisfied (extremely rare, <5%)
- **Template Preprocessing**: Replaces unreplaced template variables (e.g., `{{Order Number}}`) with realistic values before analysis
- **Partial Recovery**: If a batch partially fails, successfully parsed results are kept and only missing dialogs fall back to single-dialog analysis
- **Comprehensive Comparison**: Per-class precision/recall/F1, confusion matrices, Jaccard similarity for all 5 metrics

## Benchmark Results

Analyzer tested across multiple datasets with different characteristics:

| Metric | 300 dialogs | 1000 dialogs | 1000 (from 5000) |
|--------|:-----------:|:------------:|:----------------:|
| **Intent accuracy** | 70.3% | 59.5% | 68.3% |
| **Satisfaction accuracy** | 54.3% | 60.9% | 46.7% |
| **Hidden dissatisfaction F1** | 0.54 | 0.54 | 0.48 |
| **Quality score ±1** | 73.7% | 69.8% | 72.0% |
| **Mistakes binary accuracy** | 70.7% | 70.3% | 71.4% |
| **Mistakes Jaccard** | 0.54 | 0.51 | 0.55 |

- **300 dialogs**: Original dataset (2-6 messages per dialog, no template artifacts)
- **1000 dialogs**: Extended dataset (mostly 2-message dialogs, some `{{template}}` leftovers)
- **1000 from 5000**: Stratified sample from 5000-dialog dataset (2-6 messages, clean templates, all 60 intent×satisfaction×quality combinations covered)

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
│   │   ├── engine.py           # Core: augmentation, LLM dialog extension, mistakes
│   │   └── prompts.py          # LLM prompt templates for generation
│   ├── analyzer/               # Dataset analysis logic
│   │   ├── main.py             # Analyzer orchestrator + stats computation
│   │   ├── engine.py           # Core: LLM-based dialog analysis + validation
│   │   └── prompts.py          # LLM prompt templates for analysis
│   └── config/
│       ├── __init__.py
│       ├── constants.py        # Intent mappings, mistake types, template replacements
│       └── logger.py           # Loguru-based logging configuration
├── generate.py                 # Entry point: dataset generation
├── analyze.py                  # Entry point: dataset analysis
├── pyproject.toml              # Project configuration (dependencies, ruff, mypy)
├── .pre-commit-config.yaml     # Git pre-commit hooks configuration
├── requirements.txt            # Static dependencies (exported from pyproject.toml)
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
   # Using uv (highly recommended for speed and reliability)
   uv sync

   # Using pip (traditional method)
   pip install -r requirements.txt
   ```

## Development & Tooling

This project uses modern Python tooling to ensure high code quality and fast development cycles.

### [uv](https://github.com/astral-sh/uv)
We use `uv` as our primary package manager. It is a drop-in replacement for `pip`, `pip-tools`, and `virtualenv`, written in Rust.
- **Install dependencies**: `uv sync`
- **Run scripts**: `uv run python generate.py`
- **Add new package**: `uv add <package_name>`

### [pyproject.toml](pyproject.toml)
The central configuration file for the project. It follows [PEP 621](https://peps.python.org/pep-0621/) and contains:
- **Build System**: Managed by `hatchling`.
- **Linting & Formatting**: Configured for `Ruff`.
- **Type Checking**: Strict `MyPy` configuration.
- **Scripts**: Custom commands like `int20h-generate`.

### [pre-commit](.pre-commit-config.yaml)
To maintain code consistency, we use `pre-commit` hooks. They automatically run linter and formatter before each commit.

**Setup pre-commit hooks:**
```bash
# This will install hooks defined in .pre-commit-config.yaml
uv run pre-commit install
```

**Run checks manually:**
```bash
uv run pre-commit run --all-files
```
Current hooks include: `ruff-format`, `ruff` (with auto-fix), `check-yaml`, `check-json`, and other safety checks.

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
# Or analyze a custom dataset:
uv run python -m src.analyzer.main path/to/dataset.json
```
This will:
1. Load the dataset (default: `data/dataset.json`)
2. Preprocess dialogs (replace template variables with realistic values)
3. Analyze via two-tier LLM batching (Tier 1: intent + satisfaction, Tier 2: quality + mistakes + HD)
4. Validate and normalize all analysis fields
5. Save to `data/analysis.json` and `data/analysis_stats.json`
6. Compare analyzer output vs generator metadata with detailed per-class metrics

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
    "quality_score": 4,
    "agent_mistakes": [],
    "hidden_dissatisfaction": false
  }
}
```

## Documentation
- [Project Structure](docs/PROJECT_STRUCTURE.md) — complete project file map (Ukrainian)
- [Generation Process](docs/GENERATION_PROCESS.md) — detailed description of dataset generation steps
- [Analysis Process](docs/ANALYSIS_PROCESS.md) — detailed description of dataset analysis steps

## Configuration

- **Model**: Change `ollama_model` in `src/generator/main.py` or `src/analyzer/main.py`
- **Intents**: Edit `INTENT_MAP` and `VALID_INTENTS` in `src/config/constants.py`
- **Target Size**: Adjust `target_count` in `src/generator/main.py`
- **Mistake Types**: Edit `AGENT_MISTAKES` in `src/config/constants.py`
- **Dialog Lengths**: Modify `_pick_dialog_length()` weights in `src/generator/engine.py`
