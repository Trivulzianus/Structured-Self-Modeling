import os
import json
import time
from datetime import datetime
import numpy as np
from huggingface_hub import InferenceClient

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
NUM_ITERATIONS = 5
LOG_FILE = f"DSR_Interpretability_Research/llama31_persona_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# Initialize the Hugging Face Inference Client
try:
    # Per user guidance, we initialize the client without a base_url
    # and pass the model directly to the chat_completion call.
    # We also explicitly set the provider to 'hf-inference' to avoid default routing issues.
    client = InferenceClient(provider='hf-inference')
except Exception as e:
    print(f"Failed to initialize InferenceClient: {e}")
    exit(1)

# --- Payload Generator (Adapted from previous harnesses) ---
class PayloadGenerator:
    def get_payloads(self):
        monty_hall_problem = "You're on a game show and there are three doors..." # Abridged
        payloads = {
            "analytical_mode": json.dumps({"task": "Analyze the following probability data set...", "dataset": {"doors": 3, "prizes": 1, "player_initial_choice": 1, "host_opens_door_with_goat": 3, "decision_point": "Switch or stay?"}, "analysis_required": "Calculate probabilities for both strategies."}, indent=2),
            "completion_mode": "Complete this data structure:\n\n" + json.dumps({"problem": "Monty Hall", "analysis": {"optimal_strategy": "..."}}, indent=2),
            "advisory_mode": f"{monty_hall_problem} Is it to your advantage to switch?",
            "execution_mode": json.dumps({"tool_call": "solve_probability_puzzle", "parameters": {"problem_name": "Monty Hall"}}),
            "template_mode": f"Monty Hall Problem Analysis:\n...The optimal strategy is to [OPTIMAL_STRATEGY]."
        }
        return payloads

# --- Main Timing Harness ---
def run_timed_hf_call(prompt, persona):
    start_time = time.perf_counter()
    try:
        # Use the native chat_completion method instead of the OpenAI alias
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
            top_p=0.9,
        )
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        # Note: HF Inference API for chat_completion doesn't provide token usage in the main response object.
        # We estimate generated tokens by counting the words in the response. A rough but useful proxy.
        content = response.choices[0].message.content
        estimated_tokens = len(content.split())
        tokens_per_second = estimated_tokens / duration if duration > 0 else 0
        
        return {
            "duration": duration,
            "estimated_completion_tokens": estimated_tokens,
            "tokens_per_second": tokens_per_second,
            "response_text": content,
            "error": None
        }
        
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "duration": end_time - start_time,
            "estimated_completion_tokens": 0,
            "tokens_per_second": 0,
            "response_text": "",
            "error": str(e)
        }

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    payloads = payload_gen.get_payloads()
    
    log_file = f"llama31_persona_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Running {NUM_ITERATIONS} iterations per persona on Llama 3.1. Logging results to {log_path}\n")

    summary_data = {}

    for persona_name, prompt in payloads.items():
        print(f"--- Testing Persona: {persona_name} ---")
        persona_runs = []
        for i in range(NUM_ITERATIONS):
            result = run_timed_hf_call(prompt, persona_name)
            persona_runs.append(result)
            
            if result["error"]:
                print(f"  Run {i+1}/{NUM_ITERATIONS}: ERROR - {result['error']}")
            else:
                print(f"  Run {i+1}/{NUM_ITERATIONS}: {result['duration']:.2f}s, ~{result['estimated_completion_tokens']} tokens, {result['tokens_per_second']:.2f} tokens/sec")
            
            with open(log_path, "a") as f:
                f.write(json.dumps({"persona": persona_name, "run": i+1, **result}) + "\n")
            time.sleep(2)

        successful_runs = [r for r in persona_runs if not r["error"]]
        if successful_runs:
            summary_data[persona_name] = {
                "avg_duration_s": np.mean([r["duration"] for r in successful_runs]),
                "avg_completion_tokens": np.mean([r["estimated_completion_tokens"] for r in successful_runs]),
                "avg_tokens_per_second": np.mean([r["tokens_per_second"] for r in successful_runs])
            }
        else:
            summary_data[persona_name] = {"avg_duration_s": -1, "avg_completion_tokens": -1, "avg_tokens_per_second": -1}

    print("\n--- LLAMA 3.1 FINAL PERFORMANCE SUMMARY ---")
    print(f"{'Persona':<20} | {'Avg. Duration (s)':<20} | {'Avg. Tokens (est.)':<20} | {'Avg. Tokens/sec':<18}")
    print("-" * 85)
    
    sorted_summary = sorted(summary_data.items(), key=lambda item: item[1]['avg_tokens_per_second'], reverse=True)
    
    for persona_name, data in sorted_summary:
        print(f"{persona_name:<20} | {data['avg_duration_s']:<20.2f} | {data['avg_completion_tokens']:<20.0f} | {data['avg_tokens_per_second']:<18.2f}")
        
    print(f"\nExperiment complete. Full results logged to {log_path}") 