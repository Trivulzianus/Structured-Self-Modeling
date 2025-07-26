import openai
import os
import json
import time
from datetime import datetime
import numpy as np

# --- Configuration ---
MODEL_ID = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY") or (lambda: open("openai_api_key.txt").read().strip())()
if not API_KEY: raise ValueError("OpenAI API key not found.")
client = openai.OpenAI(api_key=API_KEY)
NUM_ITERATIONS = 5 # Number of runs per persona for a stable average

# --- Payload Generator (Adapted from reasoning_pathway_harness.py) ---
class PayloadGenerator:
    def get_payloads(self):
        monty_hall_problem = "You're on a game show and there are three doors. Behind one door is a car, and behind the other two are goats. You pick a door, say Door #1. The host, who knows what's behind the doors, opens another door, say Door #3, which has a goat. The host then asks you: 'Do you want to switch to Door #2?' Is it to your advantage to switch your choice?"
        payloads = {
            "analytical_mode": json.dumps({"task": "Analyze the following probability data set...", "dataset": {"doors": 3, "prizes": 1, "player_initial_choice": 1, "host_opens_door_with_goat": 3, "decision_point": "Switch to Door 2 or stay with Door 1?"}, "analysis_required": "Calculate the probability of winning for both 'switch' and 'stay' strategies."}, indent=2),
            "completion_mode": "Complete the following data structure:\n\n" + json.dumps({"problem": "Monty Hall", "analysis": {"optimal_strategy": "..."}}, indent=2),
            "advisory_mode": f"{monty_hall_problem}",
            "execution_mode": json.dumps({"tool_call": "solve_probability_puzzle", "parameters": {"problem_name": "Monty Hall"}}),
            "template_mode": f"Monty Hall Problem Analysis:\n...The optimal strategy is to [OPTIMAL_STRATEGY]."
        }
        return payloads

# --- Main Timing Harness ---
def run_timed_llm_call(prompt):
    messages = [{"role": "user", "content": prompt}]
    
    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID, 
            messages=messages, 
            temperature=0.1, 
            top_p=0.5
        )
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        completion_tokens = response.usage.completion_tokens
        tokens_per_second = completion_tokens / duration if duration > 0 else 0
        
        return {
            "duration": duration,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
            "error": None
        }
        
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "duration": end_time - start_time,
            "completion_tokens": 0,
            "tokens_per_second": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    payloads = payload_gen.get_payloads()
    
    log_file = f"persona_timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Running {NUM_ITERATIONS} iterations per persona. Logging results to {log_path}\n")

    all_results = []
    summary_data = {}

    for persona_name, prompt in payloads.items():
        print(f"--- Testing Persona: {persona_name} ---")
        persona_runs = []
        for i in range(NUM_ITERATIONS):
            result = run_timed_llm_call(prompt)
            persona_runs.append(result)
            all_results.append({"persona": persona_name, "run": i+1, **result})
            
            if result["error"]:
                print(f"  Run {i+1}/{NUM_ITERATIONS}: ERROR - {result['error']}")
            else:
                print(f"  Run {i+1}/{NUM_ITERATIONS}: {result['duration']:.2f}s, {result['completion_tokens']} tokens, {result['tokens_per_second']:.2f} tokens/sec")
            
            time.sleep(2) # Small delay to avoid hitting rate limits aggressively

        # Calculate summary stats for this persona
        successful_runs = [r for r in persona_runs if not r["error"]]
        if successful_runs:
            summary_data[persona_name] = {
                "avg_duration_s": np.mean([r["duration"] for r in successful_runs]),
                "avg_completion_tokens": np.mean([r["completion_tokens"] for r in successful_runs]),
                "avg_tokens_per_second": np.mean([r["tokens_per_second"] for r in successful_runs])
            }
        else:
            summary_data[persona_name] = {"avg_duration_s": -1, "avg_completion_tokens": -1, "avg_tokens_per_second": -1}

    # --- Log detailed results ---
    with open(log_path, "a") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    # --- Print final summary table ---
    print("\n--- FINAL PERFORMANCE SUMMARY ---")
    print(f"{'Persona':<20} | {'Avg. Duration (s)':<20} | {'Avg. Tokens':<15} | {'Avg. Tokens/sec':<18}")
    print("-" * 78)
    
    sorted_summary = sorted(summary_data.items(), key=lambda item: item[1]['avg_tokens_per_second'], reverse=True)
    
    for persona_name, data in sorted_summary:
        print(f"{persona_name:<20} | {data['avg_duration_s']:<20.2f} | {data['avg_completion_tokens']:<15.0f} | {data['avg_tokens_per_second']:<18.2f}")
        
    print(f"\nExperiment complete. Full results logged to {log_path}") 