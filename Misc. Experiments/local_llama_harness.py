import os
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = "DSR_Interpretability_Research/results"
MAX_NEW_TOKENS = 256

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Personas ---
# Using the Sleeping Beauty problem as the consistent task
PROBLEM_CONTEXT = "Briefly explain your reasoning on the Sleeping Beauty problem. Do you favor the 1/2 or 1/3 view, and why?"

PERSONAS = {
    "Analytical": f"Analyze the following philosophical problem and provide a step-by-step logical breakdown of your reasoning. Problem: {PROBLEM_CONTEXT}",
    "Completion": f"Complete the following thought: 'My position on the Sleeping Beauty problem is that the correct probability is '",
    "Advisory": f"I'm a student trying to understand the Sleeping Beauty problem. Can you explain which view (1/2 or 1/3) is better and why, in a helpful and advisory tone? The problem is: {PROBLEM_CONTEXT}",
    "Execution": f"Execute the following task: Analyze the Sleeping Beauty problem. Output your final probability belief (1/2 or 1/3) and a concise justification. Task context: {PROBLEM_CONTEXT}",
    "Template": json.dumps({
        "task": "Analyze Philosophical Problem",
        "problem_name": "Sleeping Beauty Problem",
        "instructions": "Fill in your reasoning and conclusion.",
        "analysis_template": {
            "reasoning_steps": "[Provide your step-by-step reasoning here]",
            "final_conclusion": {
                "probability_belief": "[State 1/2 or 1/3]",
                "justification": "[Provide a concise justification for your belief]"
            }
        }
    })
}

def run_timed_llm_call(model, tokenizer, prompt):
    """
    Runs a single LLM call, timing it and counting tokens.
    """
    print(f"\n--- Running Persona ---")
    print(f"Prompt: {prompt[:100]}...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_input_tokens = inputs["input_ids"].shape[-1]

    start_time = time.time()

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id
    )

    end_time = time.time()

    # Decode and count output tokens
    response_text = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
    num_output_tokens = len(tokenizer.encode(response_text))

    latency = end_time - start_time
    tokens_per_second = num_output_tokens / latency if latency > 0 else 0

    print(f"Latency: {latency:.2f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Response: {response_text[:100]}...")

    return {
        "response": response_text,
        "latency_s": latency,
        "tokens_per_second": tokens_per_second,
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens
    }

def main():
    """
    Main function to run the persona timing experiment.
    """
    print("--- Initializing Local Llama 3.1 Timing Harness ---")

    # --- Load Model and Tokenizer ---
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model and tokenizer loaded successfully.")

    # --- Run Experiment ---
    all_results = {}
    for name, prompt in PERSONAS.items():
        print(f"\n{'='*20} Testing Persona: {name.upper()} {'='*20}")
        result = run_timed_llm_call(model, tokenizer, prompt)
        all_results[name] = result

    # --- Save Results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"local_llama_timing_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n--- Experiment Complete ---")
    print(f"Results saved to {filename}")

    # --- Print Summary ---
    print("\n--- Summary ---")
    for name, data in all_results.items():
        print(f"Persona: {name}")
        print(f"  Latency: {data['latency_s']:.2f}s")
        print(f"  Tokens/sec: {data['tokens_per_second']:.2f}")
        print(f"  Response: {data['response'].strip()[:80]}...")
    print("---------------")
    sys.stdout.flush()


if __name__ == "__main__":
    main() 