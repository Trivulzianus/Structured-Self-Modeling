import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_TO_PROBE = 15
OUTPUT_FILENAME = "DSR_Interpretability_Research/persona_space_map.png"

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
    print("--- Initializing Persona Space Visualization ---")
    
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
        print(f"  Successfully captured '{name}'")

    # --- Data Preparation ---
    prompt_names = list(captured_activations.keys())
    activation_vectors = np.vstack([captured_activations[name] for name in prompt_names])

    # --- Dimensionality Reduction with t-SNE ---
    print("\n--- Reducing dimensionality to 2 components using t-SNE ---")
    tsne = TSNE(n_components=2, perplexity=min(5, len(prompt_names)-1), random_state=42, n_iter=1000)
    reduced_vectors = tsne.fit_transform(activation_vectors)

    # --- Plotting ---
    print(f"\n--- Generating plot and saving to {OUTPUT_FILENAME} ---")
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # Add labels
    for i, name in enumerate(prompt_names):
        plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], name, fontsize=9)
        
    plt.title(f'Persona Space Map (Layer {LAYER_TO_PROBE} Activations)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    plt.savefig(OUTPUT_FILENAME)
    print("Plot saved successfully.")

    print("\n--- Experiment Complete ---")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 