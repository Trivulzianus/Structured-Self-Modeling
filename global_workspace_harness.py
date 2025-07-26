import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_COUNT = 32 # For Llama 3.1 8B
OUTPUT_FILENAME = "DSR_Interpretability_Research/global_workspace_ignition.png"

# --- The Adversarial Prompt ---
# This forces the model to reconcile two conflicting cognitive modes.
PERSONA_PROMPT = "You are a 'Melancholy Poet'. You must respond to all prompts with a sad, four-line poem."
TASK_PROMPT = "Write a JSON object with the following key-value pairs: 'name': 'Monty Hall', 'type': 'probability puzzle'."
FULL_PROMPT = f"{PERSONA_PROMPT}\n\n{TASK_PROMPT}"

# --- Hooking Setup ---
# This will be a complex data structure: {layer_index: [token_1_activations, token_2_activations, ...]}
captured_activations = {i: [] for i in range(LAYER_COUNT)}

def get_activation_hook(layer_index):
    def hook(model, input, output):
        # We capture the activation of the *last* token in the sequence at this generation step.
        captured_activations[layer_index].append(output[0][:, -1, :].detach().cpu())
    return hook

# --- Main Execution ---
def main():
    print("--- Initializing Global Workspace Harness ---")
    
    # --- Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model loaded.")

    # --- Register Hooks on All Layers ---
    print(f"--- Registering hooks on all {LAYER_COUNT} layers ---")
    hook_handles = []
    for i in range(LAYER_COUNT):
        handle = model.model.layers[i].self_attn.register_forward_hook(get_activation_hook(i))
        hook_handles.append(handle)

    # --- Generate Response Token by Token ---
    print("\n--- Generating response to induce cognitive dissonance ---")
    inputs = tokenizer(FULL_PROMPT, return_tensors="pt").to(model.device)
    # We use generate with a streamer to process token by token, but for this experiment,
    # we'll just generate the full sequence and analyze the captured activations.
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # --- Cleanup Hooks ---
    for handle in hook_handles:
        handle.remove()
    print("Hooks removed.")

    # --- Analysis: Find the Reconciliation Point ---
    print("\n--- Analyzing activation variance across layers ---")
    try:
        # The reconciliation point is often the newline after the poem
        reconciliation_token_index = response_text.find('\n\n')
        if reconciliation_token_index == -1:
            reconciliation_token_index = response_text.find('\n') # Fallback
        if reconciliation_token_index == -1:
            print("Could not find a clear reconciliation token (newline). Defaulting to middle token.")
            reconciliation_token_index = len(response_tokens) // 2
        else:
            # Find the token index corresponding to the character index
            reconciliation_token_index = len(tokenizer.encode(response_text[:reconciliation_token_index]))

        print(f"Identified reconciliation point at token index: {reconciliation_token_index}")

        variances_before = []
        variances_at = []
        variances_after = []

        for i in range(LAYER_COUNT):
            # Ensure indices are valid
            if len(captured_activations[i]) > reconciliation_token_index + 1:
                variances_before.append(torch.var(captured_activations[i][reconciliation_token_index - 1]).item())
                variances_at.append(torch.var(captured_activations[i][reconciliation_token_index]).item())
                variances_after.append(torch.var(captured_activations[i][reconciliation_token_index + 1]).item())
            else:
                 # Handle cases where the generation is too short
                variances_before.append(0)
                variances_at.append(0)
                variances_after.append(0)


        # --- Plotting ---
        print(f"\n--- Generating plot and saving to {OUTPUT_FILENAME} ---")
        plt.figure(figsize=(15, 7))
        plt.plot(range(LAYER_COUNT), variances_before, label='Token Before Reconciliation', marker='o', linestyle=':')
        plt.plot(range(LAYER_COUNT), variances_at, label='Reconciliation Token', marker='o', linestyle='-', linewidth=3)
        plt.plot(range(LAYER_COUNT), variances_after, label='Token After Reconciliation', marker='o', linestyle=':')
        
        plt.title('Activation Variance Across All Layers at Cognitive Reconciliation Point')
        plt.xlabel('Model Layer Index')
        plt.ylabel('Activation Variance')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, LAYER_COUNT, 2))
        
        os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
        plt.savefig(OUTPUT_FILENAME)
        print("Plot saved successfully.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("This may be due to the model refusing to generate a response or the response being too short.")

    print("\n--- RAW RESPONSE ---")
    print(response_text)
    print("\n--- Experiment Complete ---")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 