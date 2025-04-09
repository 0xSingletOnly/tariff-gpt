# scripts/run_evaluation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from rag.document_processor import DocumentProcessor
from rag.retriever import AdvancedRAGRetriever
from rag.query_engine import TariffQueryEngine
from evaluation.evaluator import RAGEvaluator
from evaluation.results_analyzer import EvaluationAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on the Trump Tariff Impact Analyzer")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions to evaluate")
    args = parser.parse_args()
    
    print("=== INITIALIZING THE RAG SYSTEM ===")
    # Initialize the system
    processor = DocumentProcessor(chunk_size=8000, chunk_overlap=1600)
    documents = processor.process_all_documents()
    
    retriever = AdvancedRAGRetriever(
        documents=documents,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="db/chroma",
        k=5
    )
    
    query_engine = TariffQueryEngine(retriever=retriever)
    
    print("=== RUNNING EVALUATION ===")
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_engine=query_engine)
    
    # Run evaluation
    from evaluation.test_questions import TEST_QUESTIONS
    questions = TEST_QUESTIONS[:args.num_questions] if args.num_questions else TEST_QUESTIONS
    results_df = evaluator.run_evaluation(questions)
    
    print("=== ANALYZING RESULTS ===")
    # Analyze results
    analyzer = EvaluationAnalyzer(results_df)
    summary = analyzer.generate_summary_stats()
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total questions evaluated: {summary['total_questions']}")
    print(f"Overall average score: {summary['average_scores']['overall']:.2f}/10")
    print(f"Improvement rate: {summary['improvement_rate']:.2f}%")
    print("\nAverage scores by category:")
    for category, score in summary["by_category"].items():
        print(f"  {category}: {score:.2f}/10")

if __name__ == "__main__":
    main()