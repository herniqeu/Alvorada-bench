import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from evaluate import BenchmarkEvaluator

load_dotenv()

def load_data(source: str = "sample", hf_dataset: Optional[str] = None) -> pd.DataFrame:
    if source == "sample":
        return pd.read_csv(Path(__file__).parent.parent / "data" / "samples" / "questions_data_sample.csv")
    elif source == "huggingface":
        from datasets import load_dataset
        dataset = load_dataset(hf_dataset or "Alvorada-bench", "questions")
        return dataset['train'].to_pandas()
    else:
        return pd.read_csv(source)

def filter_questions(df: pd.DataFrame, **filters) -> pd.DataFrame:
    for key, value in filters.items():
        if value and key in df.columns:
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
    return df

def main():
    parser = argparse.ArgumentParser(description="Run Alvorada Benchmark Evaluation")
    parser.add_argument("--model", default="gpt-5", help="Model to use via LiteLLM")
    parser.add_argument("--prompt", default="zero_shot", choices=["zero_shot", "chain_of_thought", "role_playing"])
    parser.add_argument("--source", default="sample", choices=["sample", "huggingface", "custom"])
    parser.add_argument("--data-path", help="Path to custom data file")
    parser.add_argument("--output", help="Output file path (default: results/TIMESTAMP_MODEL.json)")
    parser.add_argument("--subjects", nargs="+", help="Filter by subjects")
    parser.add_argument("--exams", nargs="+", help="Filter by exam names")
    parser.add_argument("--years", nargs="+", type=int, help="Filter by exam years")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    if args.source == "custom" and not args.data_path:
        parser.error("--data-path required when using custom source")
    
    df = load_data(args.source, args.data_path if args.source == "custom" else None)
    
    filters = {}
    if args.subjects:
        filters['subject'] = args.subjects
    if args.exams:
        filters['exam_name'] = args.exams
    if args.years:
        filters['exam_year'] = args.years
    
    df = filter_questions(df, **filters)
    
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.random_seed)
    
    print(f"Evaluating {len(df)} questions with {args.model} using {args.prompt} prompt")
    
    evaluator = BenchmarkEvaluator(model=args.model, prompt_template=args.prompt)
    results = evaluator.evaluate_batch(df)
    
    accuracy = sum(r.get('correct', False) for r in results) / len(results)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    by_subject = {}
    for r in results:
        subj = r.get('subject', 'Unknown')
        if subj not in by_subject:
            by_subject[subj] = {'correct': 0, 'total': 0}
        by_subject[subj]['total'] += 1
        if r.get('correct', False):
            by_subject[subj]['correct'] += 1
    
    print("\nAccuracy by subject:")
    for subj, stats in by_subject.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {subj}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "_").replace(":", "_")
        output_path = results_dir / f"{timestamp}_{model_name}_{args.prompt}.json"
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "prompt_template": args.prompt,
            "total_questions": len(results),
            "accuracy": accuracy,
            "filters": filters if filters else None
        },
        "results": results,
        "accuracy_by_subject": {subj: {"accuracy": stats['correct']/stats['total'], **stats} for subj, stats in by_subject.items()}
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
