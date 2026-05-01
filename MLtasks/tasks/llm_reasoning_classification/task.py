import os
import json
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models import Word2Vec

def get_task_metadata():
    """Return metadata for the LLM reasoning classification task."""
    return {
        "id": "llm_reasoning_classification",
        "description": "Text Classification Pipelines Comparison on Nemotron Dataset",
        "series": "Natural Language Processing",
        "level": 4
    }

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    
def get_device():
    """Return the device to be used (CPU for scikit-learn)."""
    # Return "cpu" by default since I am using scikit-leran libraries which are CPU-only
    return "cpu"

def make_dataloaders():
    """Load the dataset and return stratified train and test splits."""
    file_path = os.path.join(os.path.dirname(__file__), "train_with_task_type.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check required columns
    if "prompt" not in df.columns or "task_type" not in df.columns:
        raise ValueError("Dataset missing required columns: 'prompt', 'task_type'")
        
    df = df.dropna(subset=['prompt', 'task_type'])
    X = df['prompt'].values
    y = df['task_type'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return (X_train, y_train), (X_test, y_test)

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for Word2Vec text embeddings."""
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        """Train the Word2Vec model on the provided text data."""
        # Tokenize by splitting by whitespace
        sentences = [str(text).lower().split() for text in X]
        self.model = Word2Vec(
            sentences=sentences, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=1,
            seed=42
        )
        self.model.train(sentences, total_examples=len(sentences), epochs=self.epochs)
        return self

    def transform(self, X, y=None):
        """Convert text data into averaged Word2Vec sentence embeddings."""
        sentences = [str(text).lower().split() for text in X]
        vectors = []
        for sentence in sentences:
            vecs = [self.model.wv[word] for word in sentence if word in self.model.wv]
            if len(vecs) > 0:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

def build_model(pipeline_name):
    """Build and return the specified scikit-learn pipeline."""
    if pipeline_name == "Word2Vec_KNN":
        return Pipeline([
            ('w2v', Word2VecTransformer(vector_size=100)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])
    elif pipeline_name == "Word2Vec_NaiveBayes":
        return Pipeline([
            ('w2v', Word2VecTransformer(vector_size=100)),
            ('nb', GaussianNB())
        ])
    elif pipeline_name == "TFIDF_SVM":
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('svm', LinearSVC(random_state=42, dual=False))
        ])
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

def train(model, X_train, y_train):
    """Train the model and measure the training time."""
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    return model, train_time

def predict(model, X):
    """Make predictions and measure the inference time."""
    start_time = time.time()
    preds = model.predict(X)
    end_time = time.time()
    inference_time = end_time - start_time
    return preds, inference_time

def evaluate(model, X, y):
    """Evaluate the model and return a dictionary of performance metrics."""
    preds, inference_time = predict(model, X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, average='weighted', zero_division=0)),
        "recall": float(recall_score(y, preds, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y, preds, average='weighted', zero_division=0)),
        "inference_time_seconds": inference_time
    }
    return metrics

def save_artifacts(results_dict, output_dir=None):
    """Save metrics to JSON and generate comparison plots."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    pipelines = list(results_dict.keys())
    
    # Metrics Comparison (Accuracy, Precision, Recall, F1)
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = {m: [results_dict[p]['test_metrics'][m] for p in pipelines] for m in metrics_names}
    
    x = np.arange(len(pipelines))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(metrics_names):
        ax.bar(x + i*width - width*1.5, metric_values[m], width, label=m.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison across Pipelines')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    # Speed Comparison (Training and Inference Time)
    train_times = [results_dict[p]['train_time_seconds'] for p in pipelines]
    inf_times = [results_dict[p]['test_metrics']['inference_time_seconds'] for p in pipelines]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(pipelines, train_times, color='skyblue')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time')
    ax1.set_xticks(range(len(pipelines)))
    ax1.set_xticklabels(pipelines, rotation=15)
    
    ax2.bar(pipelines, inf_times, color='salmon')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Inference Time (Test Set)')
    ax2.set_xticks(range(len(pipelines)))
    ax2.set_xticklabels(pipelines, rotation=15)
    
    plt.tight_layout()
    speed_plot_path = os.path.join(output_dir, "speed_comparison.png")
    plt.savefig(speed_plot_path)
    plt.close()

    # Trade-off Scatter Plot (F1 Score vs Inference Time)
    f1_scores = [results_dict[p]['test_metrics']['f1_score'] for p in pipelines]
    
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(pipelines):
        plt.scatter(inf_times[i], f1_scores[i], s=100, label=p)
        plt.annotate(p, (inf_times[i], f1_scores[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel('Inference Time (seconds) - Lower is better')
    plt.ylabel('F1 Score - Higher is better')
    plt.title('Trade-off: Predictive Performance vs. Inference Speed')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    tradeoff_plot_path = os.path.join(output_dir, "tradeoff_analysis.png")
    plt.savefig(tradeoff_plot_path)
    plt.close()

    # Trade-off Scatter Plot (F1 Score vs Training Time)
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(pipelines):
        plt.scatter(train_times[i], f1_scores[i], s=100, label=p)
        plt.annotate(p, (train_times[i], f1_scores[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel('Training Time (seconds) - Lower is better')
    plt.ylabel('F1 Score - Higher is better')
    plt.title('Trade-off: Predictive Performance vs. Training Speed')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    train_tradeoff_plot_path = os.path.join(output_dir, "training_tradeoff_analysis.png")
    plt.savefig(train_tradeoff_plot_path)
    plt.close()
    
    print(f"Artifacts saved to {output_dir}: metrics.json, metrics_comparison.png, speed_comparison.png, tradeoff_analysis.png, training_tradeoff_analysis.png")

if __name__ == '__main__':
    set_seed()
    get_device()
    
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = make_dataloaders()
    
    pipelines_to_test = ["Word2Vec_KNN", "Word2Vec_NaiveBayes", "TFIDF_SVM"]
    all_results = {}
    
    for pipeline_name in pipelines_to_test:
        print(f"\n--- Running Pipeline: {pipeline_name} ---")
        model = build_model(pipeline_name)
        
        print("Training...")
        model, train_time = train(model, X_train, y_train)
        
        print("Evaluating on train data...")
        train_metrics = evaluate(model, X_train, y_train)
        
        print("Evaluating on test data...")
        test_metrics = evaluate(model, X_test, y_test)
        
        print(f"Train F1: {train_metrics['f1_score']:.4f}")
        print(f"Test F1:   {test_metrics['f1_score']:.4f}")
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {test_metrics['inference_time_seconds']:.2f}s")
        
        all_results[pipeline_name] = {
            "train_time_seconds": train_time,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
        
    print("\nSaving artifacts...")
    save_artifacts(all_results)
    
    # Assert quality thresholds
    print("\nChecking quality thresholds...")
    best_f1 = max([res['test_metrics']['f1_score'] for res in all_results.values()])
    print(f"Best Test F1: {best_f1:.4f}")
    
    if best_f1 < 0.2:
        print("ERROR: Best test F1-score is below 0.2 threshold. Models failed to learn.")
        sys.exit(1)
        
    print("All quality thresholds met. Success.")
    sys.exit(0)
