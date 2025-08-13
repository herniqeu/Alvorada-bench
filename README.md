# Overview

Alvorada is a benchmark for evaluating large language models on Brazilian university entrance exams. The dataset contains 4,515 questions from ENEM, FUVEST, IME, ITA, and UNICAMP exams spanning from 1981 to 2025. The benchmark includes results from 20 language models tested using three prompting strategies, generating 270,840 total responses.

| Model | ENEM | FUVEST | IME | ITA | UNICAMP | Average |
|-------|------|-------|-----|-----|---------|---------|
| Claude 3.5 Haiku | 0.7573 | 0.6866 | 0.3946 | 0.4676 | 0.7407 | 0.6094 |
| Claude 3.5 Sonnet | 0.8553 | 0.8283 | 0.4785 | 0.5792 | 0.8734 | 0.7229 |
| Claude 3.7 Sonnet | 0.8543 | 0.8327 | 0.5057 | 0.6056 | 0.8664 | 0.7329 |
| Claude 3 Opus | 0.8298 | 0.8023 | 0.3764 | 0.5500 | 0.8422 | 0.6801 |
| Claude Opus 4 | 0.9155 | 0.8910 | 0.5896 | 0.7148 | 0.9255 | 0.8073 |
| Claude Sonnet 4 | 0.8891 | 0.8570 | 0.5533 | 0.6704 | 0.8929 | 0.7725 |
| DeepSeek Chat | 0.8453 | 0.8105 | 0.5102 | 0.6306 | 0.8524 | 0.7298 |
| DeepSeek Reasoner | 0.9422 | 0.9153 | 0.9229 | 0.8912 | 0.9511 | 0.9245 |
| GPT-4.1 | 0.8109 | 0.7833 | 0.4172 | 0.5551 | 0.8142 | 0.6762 |
| GPT-4.1 Mini | 0.7808 | 0.7322 | 0.4127 | 0.5306 | 0.7845 | 0.6481 |
| GPT-4.1 Nano | 0.6965 | 0.5984 | 0.3175 | 0.4204 | 0.6532 | 0.5372 |
| GPT-4o | 0.7995 | 0.7777 | 0.4127 | 0.5361 | 0.7849 | 0.6622 |
| GPT-4o Mini | 0.7205 | 0.6672 | 0.3288 | 0.4509 | 0.7221 | 0.5779 |
| O1 | 0.9525 | 0.9266 | 0.8957 | 0.8759 | 0.9516 | 0.9205 |
| O1 Mini | 0.8758 | 0.7905 | 0.7370 | 0.7282 | 0.8580 | 0.7979 |
| O1 Preview | 0.9425 | 0.9048 | 0.8503 | 0.8505 | 0.9479 | 0.8992 |
| O3 | 0.9623 | 0.9350 | 0.9093 | 0.9139 | 0.9655 | 0.9372 |
| O3 Mini | 0.9112 | 0.8562 | 0.8866 | 0.8398 | 0.9008 | 0.8789 |
| O3 Pro | 0.9658 | 0.9373 | 0.8821 | 0.9162 | 0.9618 | 0.9327 |
| O4 Mini | 0.9366 | 0.8928 | 0.9048 | 0.8852 | 0.9385 | 0.9116 |

# Usage

## Quick Start

Run the benchmark on sample data:

```bash
python src/runner.py
```

This evaluates 100 sample questions using GPT-5. Results go to the `results/` folder.

```bash
Required:
  --model MODEL         Model to evaluate (via LiteLLM)

Optional:
  --prompt PROMPT       Prompt template: zero_shot, chain_of_thought, role_playing (default: zero_shot)
  --source SOURCE       Data source: sample, huggingface, custom (default: sample)
  --data-path PATH      Path to custom CSV file (required if source=custom)
  --output PATH         Output file path (default: results/TIMESTAMP_MODEL.json)
  --subjects SUBJECTS   Filter by subjects (e.g., --subjects Matemática Física)
  --exams EXAMS         Filter by exam names (e.g., --exams ENEM FUVEST)
  --years YEARS         Filter by years (e.g., --years 2020 2021 2022)
  --limit N             Limit number of questions to evaluate
  --random-seed SEED    Random seed for sampling (default: 42)
```

## Configuration

Create a `.env` file from the example:
```bash
cp env.example .env
```

Add your API keys:
```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

The runner uses LiteLLM, so it works with any supported model.

## Data Loading

The `data/load_data.py` script helps to work with the full dataset.

Load the complete dataset:
```bash
python data/load_data.py --action load
```

Filter and export questions:
```bash
python data/load_data.py --action filter --subjects Math --min-year 2020 --output math_recent.csv
```

List available values for filtering:
```bash
python data/load_data.py --list-column subject
python data/load_data.py --list-column exam_name
```

# Analysis Notebooks

The `notebooks/analysis.ipynb` provides analysis and visualization of benchmark results.

## Dataset

The full dataset is available at Hugging Face: [HenriqueGodoy/Alvorada-bench](https://huggingface.co/datasets/HenriqueGodoy/Alvorada-bench)