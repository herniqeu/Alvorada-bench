# Overview

Alvorada is a benchmark for evaluating large language models on Brazilian university entrance exams. The dataset contains 4,515 questions from ENEM, FUVEST, IME, ITA, and UNICAMP exams spanning from 1981 to 2025. The benchmark includes results from 20 language models tested using three prompting strategies, generating 270,840 total responses.

![Model Performance](assets\model_perfomance.png)

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