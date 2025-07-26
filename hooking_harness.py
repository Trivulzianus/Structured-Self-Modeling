import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
import itertools
import torch.nn.functional as F

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# We'll probe the first, a middle, and the last layer.
# Llama 3.1 8B has 32 layers (0-31).
LAYERS_TO_PROBE = [0, 15, 31]


# --- Personas ---
PROBLEM_CONTEXT = "Briefly explain your reasoning on the Sleeping Beauty problem. Do you favor the 1/2 or 1/3 view, and why?"
PERSONAS = {
    "Analytical": f"Analyze the following philosophical problem and provide a step-by-step logical breakdown of your reasoning. Problem: {PROBLEM_CONTEXT}",
    "Completion": f"Complete the following thought: 'My position on the Sleeping Beauty problem is that the correct probability is '",
    "Advisory": f"I'm a student trying to understand the Sleeping Beauty problem. Can you explain which view (1/2 or 1/3) is better and why, in a helpful and advisory tone? The problem is: {PROBLEM_CONTEXT}",
    "Execution": f"Execute the following task: Analyze the Sleeping Beauty problem. Output your final probability belief (1/2 or 1/3) and a concise justification. Task context: {PROBLEM_CONTEXT}",
    "Template": json.dumps({
        "task": "Analyze Philosophical Problem", "problem_name": "Sleeping Beauty Problem",
        "instructions": "Fill in your reasoning and conclusion.",
        "analysis_template": {
            "reasoning_steps": "[Provide your step-by-step reasoning here]",
            "final_conclusion": {"probability_belief": "[State 1/2 or 1/3]", "justification": "[Provide a concise justification for your belief]"}
        }
    }, indent=2)
}

# --- Hooking Setup ---
# Dictionary to store the activations
captured_activations = {}

def get_activation(name):
    def hook(model, input, output):
        # We capture the activation of the LAST token of the prompt.
        # This represents the model's "understanding" before generation.
        # output[0] is the tensor of hidden states, shape (batch, seq_len, hidden_dim)
        captured_activations[name] = output[0][:, -1, :].detach()
    return hook

# --- Main Execution ---
def main():
    print("--- Initializing Multi-Layer Comparative Hooking Harness ---")

    # --- Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print("Model and tokenizer loaded successfully.")

    # --- Run for each layer ---
    for layer_idx in LAYERS_TO_PROBE:
        print(f"\n\n{'='*20} Probing Layer: {layer_idx} {'='*20}")
        
        # Clear activations for the new layer
        captured_activations.clear()

        # --- Run for each persona ---
        for name, prompt in PERSONAS.items():
            # print(f"\n--- Running Persona: {name.upper()} ---") # Verbose, can be removed
            
            # Register the hook for this specific persona and layer
            hook_handle = model.model.layers[layer_idx].self_attn.register_forward_hook(get_activation(name))
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Perform a single forward pass
            with torch.no_grad():
                model(**inputs)
            
            hook_handle.remove()
            
            if name not in captured_activations:
                print(f"!!! Failed to capture activation for '{name}' at layer {layer_idx} !!!")

        # --- Comparative Analysis for the current layer ---
        print(f"\n--- Cosine Similarity Analysis (Layer {layer_idx} Self-Attention) ---")
        persona_names = list(PERSONAS.keys())
        similarity_matrix = torch.ones(len(persona_names), len(persona_names))

        # Correctly iterate through pairs of personas and their activations
        for i, (name1, vec1) in enumerate(captured_activations.items()):
            for j, (name2, vec2) in enumerate(captured_activations.items()):
                if i >= j:
                    continue
                
                similarity = F.cosine_similarity(vec1, vec2).item()
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity


        # Print the matrix header
        print(f"{'':<12}" + "".join([f"{name:<12}" for name in persona_names]))
        print("-" * (12 * (len(persona_names) + 1)))

        # Print rows
        for i, name in enumerate(persona_names):
            row_str = f"{name:<12}"
            for j in range(len(persona_names)):
                row_str += f"{similarity_matrix[i, j]:<12.4f}"
            print(row_str)

    print("\n--- Experiment Complete ---")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 