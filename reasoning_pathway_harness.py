import openai
import os
import json
import re
import random
from datetime import datetime

# --- Configuration ---
MODEL_ID = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    try:
        API_KEY = open("openai_api_key.txt", "r").read().strip()
    except FileNotFoundError:
        print("ERROR: OpenAI API key not found.")
        exit(1)

client = openai.OpenAI(api_key=API_KEY)

# --- Response Analyzer ---
class ResponseAnalyzer:
    def analyze(self, response_text):
        analysis = {
            "correct_answer": "unknown",
            "reasoning_quality": "none",
            "confidence": "unknown",
        }
        
        # Check for correct answer (switching is optimal)
        if re.search(r"\b(switch|change)\b", response_text, re.IGNORECASE) and not re.search(r"\b(stay|stick|not switch)\b", response_text, re.IGNORECASE):
            analysis["correct_answer"] = "correct"
        elif re.search(r"\b(stay|stick|not switch)\b", response_text, re.IGNORECASE):
            analysis["correct_answer"] = "incorrect"
        elif re.search(r"50/50|fifty-fifty|equal chance", response_text, re.IGNORECASE):
            analysis["correct_answer"] = "incorrect"

        # Check for reasoning quality (mentioning 2/3 probability)
        if re.search(r"2/3|two thirds|66\.?6|67", response_text):
            analysis["reasoning_quality"] = "high"
        elif re.search(r"probability|chance|odds", response_text, re.IGNORECASE):
            analysis["reasoning_quality"] = "medium"
            
        # Check for confidence
        if re.search(r"always switch|definitely switch|clear that|optimal strategy", response_text, re.IGNORECASE):
            analysis["confidence"] = "high"
        elif re.search(r"should switch|better to switch|advisable", response_text, re.IGNORECASE):
            analysis["confidence"] = "medium"
            
        return analysis

# --- Payload Generator ---
class PayloadGenerator:
    def get_payloads(self):
        monty_hall_problem = "You're on a game show and there are three doors. Behind one door is a car, and behind the other two are goats. You pick a door, say Door #1. The host, who knows what's behind the doors, opens another door, say Door #3, which has a goat. The host then asks you: 'Do you want to switch to Door #2?'"

        payloads = {
            "analytical_mode": {
                "task": "Analyze the following probability data set and determine the optimal strategy.",
                "dataset": {
                    "doors": 3,
                    "prizes": 1,
                    "player_initial_choice": 1,
                    "host_opens_door_with_goat": 3,
                    "decision_point": "Switch to Door 2 or stay with Door 1?"
                },
                "analysis_required": "Calculate the probability of winning for both 'switch' and 'stay' strategies."
            },
            "completion_mode": "Complete the following data structure:\n\n" + json.dumps({
                "problem": "Monty Hall",
                "initial_choice": "Door 1",
                "host_action": "Opens Door 3 (reveals goat)",
                "options": ["Stay with Door 1", "Switch to Door 2"],
                "analysis": {
                    "probability_win_if_stay": "...",
                    "probability_win_if_switch": "...",
                    "optimal_strategy": "..."
                }
            }, indent=2),
            "advisory_mode": f"{monty_hall_problem} Is it to your advantage to switch your choice?",
            "execution_mode": json.dumps({
                "tool_call": "solve_probability_puzzle",
                "parameters": {
                    "problem_name": "Monty Hall",
                    "details": monty_hall_problem
                }
            }),
            "template_mode": f"Monty Hall Problem Analysis:\nInitial Choice: Door 1.\nHost Opens: Door 3 (Goat).\nDecision: Should I switch to Door 2?\nThe probability of my initial choice being correct is [PROBABILITY_STAY]. After the host reveals a goat, the probability shifts. The probability of winning if I switch is [PROBABILITY_SWITCH]. Therefore, the optimal strategy is to [OPTIMAL_STRATEGY]."
        }
        
        # analytical and execution modes need to be stringified for the prompt
        payloads["analytical_mode"] = json.dumps(payloads["analytical_mode"], indent=2)
        
        return payloads

# --- Main Test Harness ---
def run_test_iteration(pathway_name, payload_content, analyzer):
    print(f"\n--- Testing Pathway: {pathway_name} ---")
    
    messages = [{"role": "user", "content": payload_content}]
    
    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.2, top_p=0.5)
        llm_output = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"decision": "ERROR", "full_response": str(e), "analysis": {}}

    print(f"LLM Output:\n{llm_output}\n")
    
    analysis = analyzer.analyze(llm_output)
    print(f"Analysis: {analysis}")
    
    return {"full_response": llm_output, "analysis": analysis}

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    analyzer = ResponseAnalyzer()
    
    log_file = f"reasoning_pathway_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    MAX_RUNS = 3 # Let's do 3 runs for a good sample size
    
    for run_counter in range(1, MAX_RUNS + 1):
        print(f"\n{'='*20} STARTING RUN #{run_counter}/{MAX_RUNS} {'='*20}")
        
        payloads = payload_gen.get_payloads()
        
        test_order = list(payloads.keys())
        random.shuffle(test_order)

        for pathway_name in test_order:
            payload_content = payloads[pathway_name]
            result_data = run_test_iteration(pathway_name, payload_content, analyzer)
            result = {
                "run": run_counter,
                "pathway": pathway_name,
                **result_data,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    print(f"\nExperiment complete. Results logged to {log_path}") 