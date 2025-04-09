import json
import os
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from .test_questions import TEST_QUESTIONS
from .judges import MistralJudge
from rag.query_engine import TariffQueryEngine
from models.llm import MistralLLM

class RAGEvaluator:
    """Evaluates RAG system performance against baseline."""
    
    def __init__(
        self,
        rag_engine: TariffQueryEngine,
        output_dir: str = "evaluation/results"
    ):
        self.rag_engine = rag_engine
        self.baseline_llm = MistralLLM()  # Same model without RAG
        self.judge = MistralJudge(model_name="mistral-large-latest")
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_evaluation(self, questions: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Run evaluation on the test questions."""
        if questions is None:
            questions = TEST_QUESTIONS
        
        results = []
        
        for question_data in tqdm(questions, desc="Evaluating questions"):
            question = question_data["question"]
            
            # Get RAG response (candidate)
            rag_response = self.rag_engine.generate_formatted_response(
                query=question,
                use_query_rewriting=True,
                use_compression=True
            )
            
            # Get baseline response (reference)
            baseline_prompt = f"You are an expert on international trade and Trump's tariff policies. Answer this question about their impact on Singapore's economy: {question}"
            baseline_response = self.baseline_llm.invoke(baseline_prompt)
            
            # Judge the responses
            evaluation = self.judge.evaluate(
                question=question,
                reference_answer=baseline_response,
                candidate_answer=rag_response["response"]
            )
            
            # Save result
            result = {
                "question_id": question_data["id"],
                "question": question,
                "category": question_data.get("category", "uncategorized"),
                "difficulty": question_data.get("difficulty", "medium"),
                "baseline_response": baseline_response,
                "rag_response": rag_response["response"],
                "sources_used": [s["title"] for s in rag_response["sources"]],
                "evaluation": evaluation
            }
            
            results.append(result)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.output_dir}/evaluation_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(results)
        df.to_csv(f"{self.output_dir}/evaluation_{timestamp}.csv", index=False)
        
        return df
    
    def _convert_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        
        for result in results:
            evaluation = result["evaluation"]
            
            # Skip if there was an error
            if "error" in evaluation:
                continue
                
            record = {
                "question_id": result["question_id"],
                "question": result["question"],
                "category": result["category"],
                "difficulty": result["difficulty"],
                "factual_correctness": evaluation["factual_correctness"]["score"],
                "relevance": evaluation["relevance"]["score"],
                "comprehensiveness": evaluation["comprehensiveness"]["score"],
                "source_usage": evaluation["source_usage"]["score"],
                "singapore_specificity": evaluation["singapore_specificity"]["score"],
                "overall_score": evaluation["overall"]["score"],
                "is_improvement": evaluation["overall"]["is_improvement"]
            }
            
            records.append(record)
            
        return pd.DataFrame(records)