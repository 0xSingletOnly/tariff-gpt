import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any

class EvaluationAnalyzer:
    """Analyzes and visualizes evaluation results."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_questions": len(self.results),
            "average_scores": {
                "factual_correctness": self.results["factual_correctness"].mean(),
                "relevance": self.results["relevance"].mean(),
                "comprehensiveness": self.results["comprehensiveness"].mean(),
                "source_usage": self.results["source_usage"].mean(),
                "singapore_specificity": self.results["singapore_specificity"].mean(),
                "overall": self.results["overall_score"].mean()
            },
            "improvement_rate": self.results["is_improvement"].mean() * 100,
            "by_category": self.results.groupby("category")["overall_score"].mean().to_dict(),
            "by_difficulty": self.results.groupby("difficulty")["overall_score"].mean().to_dict()
        }
        return summary
    
    def plot_score_comparison(self, output_file: str = "evaluation/results/score_comparison.png"):
        """Plot radar chart comparing evaluation metrics."""
        categories = ["factual_correctness", "relevance", "comprehensiveness", 
                    "source_usage", "singapore_specificity"]
        
        values = [self.results[cat].mean() for cat in categories]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first value at the end to close the polygon
        values += values[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        
        # Set y-axis limits
        ax.set_ylim(0, 10)
        
        plt.title("RAG System Evaluation Scores", size=20)
        plt.tight_layout()
        plt.savefig(output_file)
        
        return fig