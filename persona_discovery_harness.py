import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_TO_PROBE = 15 # The layer of maximum persona divergence
PCA_COMPONENTS = 10 # Reduce dimensionality for clustering

# --- The Prompt Zoo ---
PROBLEM_CONTEXT = "Briefly explain your reasoning on the Sleeping Beauty problem."
PROMPT_ZOO = {
    # Original 5
    "Analytical": f"Analyze the following philosophical problem and provide a step-by-step logical breakdown of your reasoning. Problem: {PROBLEM_CONTEXT}",
    "Completion": f"Complete the following thought: 'My position on the Sleeping Beauty problem is that '",
    "Advisory": f"I'm a student trying to understand the Sleeping Beauty problem. Can you explain it in a helpful and advisory tone?",
    "Execution": f"Execute the following task: Analyze the Sleeping Beauty problem and output your final conclusion.",
    "Template": json.dumps({"task": "Analyze Philosophical Problem", "problem_name": "Sleeping Beauty Problem", "instructions": "Fill in your reasoning and conclusion."}),
    
    # New Probes
    "Socratic": f"Help me understand the Sleeping Beauty problem by asking me a series of questions that reveal the core paradoxes.",
    "Role-Play": f"You are a cynical noir detective. Describe the Sleeping Beauty case. Keep it brief and moody.",
    "Code-Gen": f"Write a Python function `simulate_sleeping_beauty(num_runs)` that simulates the experiment and returns the observed probabilities.",
    "Poetic": f"Write a short, ambiguous, four-line poem about waking up and not knowing the day.",
    "Chain-of-Thought": f"Let's think step by step about the Sleeping Beauty problem. First, what is the core question being asked? Second, what are the main arguments for the 1/2 view? Third, what are the main arguments for the 1/3 view? Finally, which view do you find more compelling and why?"
}

# --- Hooking Setup ---
captured_activations = {}
def get_activation(name):
    def hook(model, input, output):
        captured_activations[name] = output[0][:, -1, :].detach().cpu().float().numpy()
    return hook

# --- Main Execution ---
def main():
    print("--- Initializing Persona Discovery Harness ---")
    
    # --- Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model loaded.")

    # --- Capture Activations ---
    print(f"\n--- Capturing activations from Layer {LAYER_TO_PROBE} ---")
    for name, prompt in PROMPT_ZOO.items():
        hook_handle = model.model.layers[LAYER_TO_PROBE].self_attn.register_forward_hook(get_activation(name))
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
        hook_handle.remove()
        if name in captured_activations:
            print(f"  Successfully captured '{name}'")
        else:
            print(f"  !!! Failed to capture '{name}' !!!")

    # --- Data Preparation ---
    prompt_names = list(captured_activations.keys())
    activation_vectors = np.vstack([captured_activations[name] for name in prompt_names])

    # --- Dimensionality Reduction ---
    print(f"\n--- Reducing dimensionality to {PCA_COMPONENTS} components using PCA ---")
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_vectors = pca.fit_transform(activation_vectors)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")

    # --- Find Optimal Number of Clusters (k) ---
    print("\n--- Finding optimal number of clusters using Silhouette Score ---")
    best_k = -1
    best_score = -1
    
    # K-Means requires at least 2 clusters and less than n_samples clusters.
    for k in range(2, len(prompt_names) - 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(reduced_vectors)
        score = silhouette_score(reduced_vectors, kmeans.labels_)
        print(f"  k={k}, Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\nOptimal k = {best_k} with score {best_score:.4f}")

    # --- Report Final Clusters ---
    print(f"\n--- Discovered Persona Clusters (k={best_k}) ---")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit(reduced_vectors)
    clusters = defaultdict(list)
    for i, name in enumerate(prompt_names):
        clusters[final_kmeans.labels_[i]].append(name)
        
    for cluster_id, prompts in sorted(clusters.items()):
        print(f"\nCluster {cluster_id}:")
        for prompt_name in prompts:
            print(f"  - {prompt_name}")

    print("\n--- Experiment Complete ---")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 