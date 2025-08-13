import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from typing import Optional, List, Dict, Any

class AlvoradaDataLoader:
    def __init__(self, source: str = "huggingface", cache_dir: str = "data/full"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.questions_df = None
        self.responses_df = None
        self.source = source
        
    def load(self, force_download: bool = False, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        questions_path = self.cache_dir / "questions_full.csv"
        responses_path = self.cache_dir / "responses_full.csv"
        
        if not force_download and questions_path.exists() and responses_path.exists():
            if verbose:
                print(f"Using cached data from {self.cache_dir}")
            self.questions_df = pd.read_csv(questions_path)
            self.responses_df = pd.read_csv(responses_path)
            if verbose:
                print(f"Loaded: {len(self.questions_df)} questions, {len(self.responses_df)} responses")
        else:
            if verbose:
                print("Downloading from HuggingFace...")
            try:
                questions_dataset = load_dataset("HenriqueGodoy/Alvorada-bench", "questions")
                responses_dataset = load_dataset("HenriqueGodoy/Alvorada-bench", "responses")
                
                self.questions_df = questions_dataset['train'].to_pandas()
                self.responses_df = responses_dataset['train'].to_pandas()
                
                self.questions_df.to_csv(questions_path, index=False)
                self.responses_df.to_csv(responses_path, index=False)
                if verbose:
                    print(f"Cached: {len(self.questions_df)} questions, {len(self.responses_df)} responses")
            except Exception as e:
                if questions_path.exists() and responses_path.exists():
                    if verbose:
                        print(f"Download failed, using existing cache: {e}")
                    self.questions_df = pd.read_csv(questions_path)
                    self.responses_df = pd.read_csv(responses_path)
                else:
                    raise e
        
        return self.questions_df, self.responses_df
    
    def filter_questions(
        self,
        subjects: Optional[List[str]] = None,
        exams: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        exam_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        contains_text: Optional[str] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        
        if self.questions_df is None:
            self.load()
        
        df = self.questions_df.copy()
        
        if len(df) == 0:
            return df
        
        if subjects and 'subject' in df.columns:
            df = df[df['subject'].isin(subjects)]
        
        if exams and 'exam_name' in df.columns:
            df = df[df['exam_name'].isin(exams)]
        
        if years and 'exam_year' in df.columns:
            df = df[df['exam_year'].isin(years)]
        
        if exam_types and 'exam_type' in df.columns:
            df = df[df['exam_type'].isin(exam_types)]
        
        if min_year and 'exam_year' in df.columns:
            df = df[df['exam_year'] >= min_year]
        
        if max_year and 'exam_year' in df.columns:
            df = df[df['exam_year'] <= max_year]
        
        if contains_text and 'question_statement' in df.columns:
            mask = df['question_statement'].str.contains(contains_text, case=False, na=False)
            df = df[mask]
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_seed)
        
        return df
    
    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None:
            df = self.questions_df
        
        if df is None or len(df) == 0:
            return {}
        
        stats = {
            "total_questions": len(df),
            "subjects": df['subject'].value_counts().to_dict(),
            "exams": df['exam_name'].value_counts().head(10).to_dict(),
            "exam_types": df['exam_type'].value_counts().to_dict(),
            "years": {
                "min": int(df['exam_year'].min()),
                "max": int(df['exam_year'].max()),
                "distribution": df['exam_year'].value_counts().head(10).to_dict()
            },
            "questions_per_subject": df.groupby('subject').size().to_dict(),
            "avg_alternatives": (~df[['alternative_a', 'alternative_b', 'alternative_c', 'alternative_d', 'alternative_e']].isna()).sum(axis=1).mean()
        }
        return stats
    
    def export_filtered(self, df: pd.DataFrame, output_path: str, format: str = "csv"):
        output_path = Path(output_path)
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_unique_values(self, column: str) -> List:
        if self.questions_df is None:
            self.load()
        
        if column not in self.questions_df.columns:
            return []
        
        unique_values = self.questions_df[column].dropna().unique().tolist()
        return sorted(unique_values) if column != 'exam_year' else sorted(unique_values, reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Load and filter Alvorada benchmark data")
    
    parser.add_argument("--action", choices=["load", "filter", "stats", "list"], 
                       default="load", help="Action to perform")
    parser.add_argument("--force-download", action="store_true", 
                       help="Force download even if cache exists")
    
    parser.add_argument("--subjects", nargs="+", help="Filter by subjects")
    parser.add_argument("--exams", nargs="+", help="Filter by exam names")
    parser.add_argument("--years", nargs="+", type=int, help="Filter by specific years")
    parser.add_argument("--exam-types", nargs="+", help="Filter by exam types")
    parser.add_argument("--min-year", type=int, help="Minimum year")
    parser.add_argument("--max-year", type=int, help="Maximum year")
    parser.add_argument("--contains", help="Filter questions containing text")
    parser.add_argument("--sample", type=int, help="Random sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "json", "parquet"], 
                       default="csv", help="Output format")
    
    parser.add_argument("--list-column", help="List unique values for a column")
    parser.add_argument("--show-columns", action="store_true", help="Show all column names")
    
    args = parser.parse_args()
    
    loader = AlvoradaDataLoader()
    
    if args.action == "load":
        questions, responses = loader.load(force_download=args.force_download, verbose=True)
        print(f"Questions shape: {questions.shape}")
        print(f"Responses shape: {responses.shape}")
    
    elif args.action == "filter":
        loader.load()
        
        filtered_df = loader.filter_questions(
            subjects=args.subjects,
            exams=args.exams,
            years=args.years,
            exam_types=args.exam_types,
            min_year=args.min_year,
            max_year=args.max_year,
            contains_text=args.contains,
            sample_size=args.sample,
            random_seed=args.seed
        )
        
        print(f"Filtered: {len(filtered_df)} questions")
        
        if args.output:
            loader.export_filtered(filtered_df, args.output, args.format)
            print(f"Exported to {args.output}")
    
    elif args.action == "stats":
        loader.load()
        stats = loader.get_statistics()
        
        print(f"Total questions: {stats['total_questions']}")
        print(f"Subjects: {len(stats['subjects'])}")
        print(f"Year range: {stats['years']['min']}-{stats['years']['max']}")
        print(f"Exam types: {list(stats['exam_types'].keys())}")
    
    elif args.action == "list":
        if args.show_columns:
            loader.load()
            print("Columns:", list(loader.questions_df.columns))
        
        elif args.list_column:
            values = loader.get_unique_values(args.list_column)
            print(f"{args.list_column} ({len(values)} values):", values[:20])
            if len(values) > 20:
                print(f"... and {len(values) - 20} more")

if __name__ == "__main__":
    main()