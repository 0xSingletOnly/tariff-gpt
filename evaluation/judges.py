from typing import Dict, Any
import json
from langchain.prompts import PromptTemplate
from models.llm import MistralLLM

class MistralJudge:
    """Evaluates response quality using Mistral-large."""
    
    def __init__(self, model_name="mistral-large-latest"):
        self.llm = MistralLLM(model_name=model_name)
        self._setup_evaluation_prompt()
    
    def _setup_evaluation_prompt(self):
        """Set up the evaluation prompt template."""
        self.prompt = PromptTemplate(
            input_variables=["question", "reference_answer", "candidate_answer"],
            template="""You are an expert judge evaluating the quality of AI responses about Trump's tariff policies and their impact on Singapore's economy.

QUESTION: {question}

REFERENCE ANSWER (from system without RAG): 
{reference_answer}

CANDIDATE ANSWER (from system with RAG): 
{candidate_answer}

Evaluate the candidate answer compared to the reference answer on the following criteria:
1. Factual Correctness (1-10): Is the information accurate and well-supported?
2. Relevance (1-10): How directly does it address the question?
3. Comprehensiveness (1-10): How complete is the answer?
4. Source Usage (1-10): How effectively are sources cited and used?
5. Singapore Specificity (1-10): How well does it focus on Singapore's context?

For each criterion, provide:
- Score (1-10)
- Brief explanation for the score

Then provide an overall assessment with:
- Overall score (1-10)
- Summary of strengths and weaknesses
- Whether the RAG system (candidate) provides clear improvement over the base system (reference)

FORMAT YOUR RESPONSE AS JSON:
{{
  "factual_correctness": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "relevance": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "comprehensiveness": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "source_usage": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "singapore_specificity": {{
    "score": <score>,
    "explanation": "<explanation>"
  }},
  "overall": {{
    "score": <score>,
    "summary": "<summary>",
    "is_improvement": <true/false>
  }}
}}
"""
        )
    
    def evaluate(self, question: str, reference_answer: str, candidate_answer: str) -> Dict[str, Any]:
        """Evaluate answers using the Mistral judge."""
        chain = self.prompt | self.llm
        response = chain.invoke({
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer
        })

        cleaned_response = response.strip()
        # Remove ```json at the beginning and ``` at the end if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  # Remove ```
            
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
        cleaned_response = cleaned_response.strip()
        
        # Parse JSON response
        try:
            evaluation = json.loads(cleaned_response.strip())
            return evaluation
        except json.JSONDecodeError:
            # Fallback in case of parsing issues
            return {
                "error": "Failed to parse JSON response",
                "raw_response": cleaned_response
            }