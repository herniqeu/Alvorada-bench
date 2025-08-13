import pandas as pd
import json
from pathlib import Path
import litellm
from typing import Dict, Any, List, Optional

class BenchmarkEvaluator:
    def __init__(self, model: str = "gpt-5", prompt_template: str = "zero_shot"):
        self.model = model
        self.prompt_template = self._load_prompt_template(prompt_template)
    
    def _load_prompt_template(self, template_name: str) -> str:
        prompt_path = Path(__file__).parent / "prompts" / f"{template_name}.md"
        return prompt_path.read_text(encoding='utf-8')
    
    def _format_question(self, row: pd.Series) -> str:
        options = []
        for letter in ['a', 'b', 'c', 'd', 'e']:
            if pd.notna(row[f'alternative_{letter}']):
                options.append(f"{letter}) {row[f'alternative_{letter}']}")
        return self.prompt_template.format(
            question=row['question_statement'],
            options='\n'.join(options)
        )
    
    def evaluate_question(self, row: pd.Series) -> Dict[str, Any]:
        prompt = self._format_question(row)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            
            response = litellm.completion(**kwargs)
            result = json.loads(response.choices[0].message.content)
            result['correct'] = result['chosen_answer'] == row['correct_answer']
            result['question_id'] = row['question_id']
            result['subject'] = row['subject']
            result['exam_name'] = row['exam_name']
            return result
        except Exception as e:
            return {
                'question_id': row['question_id'],
                'error': str(e),
                'correct': False
            }
    
    def evaluate_batch(self, df: pd.DataFrame, progress: bool = True) -> List[Dict[str, Any]]:
        results = []
        for idx, row in df.iterrows():
            if progress:
                print(f"Evaluating {idx+1}/{len(df)}", end='\r')
            results.append(self.evaluate_question(row))
        if progress:
            print()
        return results
