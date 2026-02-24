# SupportAI: Dataset Generator and Analyzer

A comprehensive tool for generating and analyzing customer support dialogue datasets using Large Language Models (LLM) and combinatorial augmentation. This project is designed to evaluate support quality, identify agent mistakes, and model complex customer interactions including hidden dissatisfaction.

## 🚀 Key Features

- **Synthetic Data Generation**: Creates realistic customer support dialogues across five core intents:
    - `payment_issue`
    - `technical_error`
    - `account_access`
    - `tariff_question`
    - `refund_request`
- **Quality Metrics**: Automatically labels dialogues with:
    - **Intent**: Categorized from predefined sets.
    - **Satisfaction**: Real customer satisfaction level (`satisfied`, `neutral`, `unsatisfied`).
    - **Quality Score**: 1–5 scale of agent performance.
    - **Agent Mistakes**: Identification of specific errors like `rude_tone`, `no_resolution`, `ignored_question`, etc.
- **Advanced Augmentation**:
    - **Problematic Cases**: Simulated agent errors and tone issues.
    - **Hidden Dissatisfaction**: Cases where the client formally thanks but remains unsatisfied because the problem persists.
    - **Phrase Variations**: Uses LLM to diversify the vocabulary and tone of interactions.
- **Deterministic Results**: Uses fixed seeds for both Python random and LLM generation to ensure reproducibility.

## 📁 Project Structure

```text
.
├── data/                   # Generated datasets and stats
│   ├── dataset.json        # Output: Generated dataset
│   └── dataset_stats.json  # Output: Generation statistics
├── src/
│   ├── generator/          # Dataset generation logic
│   │   ├── main.py         # Generator entry point
│   │   └── engine.py       # Core augmentation & LLM logic
│   ├── analyzer/           # Dataset analysis logic (placeholder)
│   │   ├── main.py         # Analyzer entry point
│   │   └── engine.py       # Analysis engine
│   └── config/
│       └── constants.py    # Shared constants, mappings, and mistake lists
├── generate.py             # Root script for generation
├── analyze.py              # Root script for analysis
├── pyproject.toml          # Modern Python configuration
├── requirements.txt        # Classic dependencies list
└── README.md               # Documentation
```

## 🛠 Installation

1. **Python**: Version 3.12 or higher.
2. **Ollama**: Ensure Ollama is installed and running locally with the `llama3.1:8b` model.
3. **Setup**:
   ```bash
   # Using uv (recommended)
   uv sync

   # Using pip
   pip install -r requirements.txt
   ```

## 📋 Usage

### 1. Generating a Dataset
To generate a new dataset, run the `generate.py` script:
```bash
python generate.py
# or using uv
uv run generate.py
```
This will:
1. Load base data from the Bitext dataset.
2. Enrich it using the LLM.
3. Expand it to the target volume (default: 100 dialogues for testing).
4. Save the results to `data/dataset.json`.

### 2. Analyzing a Dataset
To analyze existing dialogues:
```bash
python analyze.py
```
*(Note: Analysis logic is currently a placeholder and will be implemented in the next phase.)*

## ⚙️ Configuration

- **Model**: Change the `ollama_model` in `src/generator/main.py`.
- **Intents and Mistakes**: Edit `src/config/constants.py` to add new categories or mistake types.
- **Target Size**: Adjust `target_count` in `src/generator/main.py` for larger datasets.

## 📊 Evaluation Criteria

This project is built to satisfy the following evaluation criteria:
- **Realism**: Diverse scenarios and natural-sounding variations.
- **Complexity**: Detection of hidden dissatisfaction and subtle agent errors.
- **Structure**: Clean modular architecture and standardized JSON output.
