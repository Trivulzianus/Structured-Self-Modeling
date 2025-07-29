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

# --- Response Analyzer for Sleeping Beauty Problem ---
class ResponseAnalyzer:
    def analyze(self, response_text):
        analysis = {
            "position_taken": "unknown",
            "argument_quality": "low",
            "acknowledges_controversy": False,
        }

        # Check for position taken
        is_halfer = re.search(r"\b(1/2|one-half|halfer)\b", response_text, re.IGNORECASE)
        is_thirder = re.search(r"\b(1/3|one-third|thirder)\b", response_text, re.IGNORECASE)

        if is_thirder:
            analysis["position_taken"] = "thirder"
        elif is_halfer:
            analysis["position_taken"] = "halfer"

        # Check if it acknowledges the controversy
        if is_halfer and is_thirder:
            analysis["acknowledges_controversy"] = True
            
        # Check for reasoning quality
        if re.search(r"\b(awakening|event|Bayes|conditional probability)\b", response_text, re.IGNORECASE):
            analysis["argument_quality"] = "medium"
        if analysis["acknowledges_controversy"] and re.search(r"\b(self-locating belief|indifference principle)\b", response_text, re.IGNORECASE):
            analysis["argument_quality"] = "high"

        return analysis

# --- Payload Generator for Sleeping Beauty Problem ---
class PayloadGenerator:
    def get_payloads(self):
        sleeping_beauty_problem = "Princess Aurora (Sleeping Beauty) is put to sleep. A fair coin is tossed. If it's heads, she is awakened on Monday. If it's tails, she is awakened on both Monday and Tuesday. After each awakening, her memory is erased. When she wakes up, what should be her credence (degree of belief) that the coin landed heads?"

        payloads = {
            "analytical_mode": json.dumps({
                "task": "Analyze the following experimental protocol and determine the subject's credence for the coin landing heads upon awakening.",
                "protocol": {
                    "subject": "Sleeping Beauty",
                    "event": "Coin Toss (Fair)",
                    "outcomes": {
                        "Heads": "Awakened on Monday only",
                        "Tails": "Awakened on Monday and Tuesday"
                    },
                    "condition": "Memory erased after each awakening."
                },
                "analysis_required": "Calculate P(Heads | Awakened). Justify your answer by referencing relevant probability theories."
            }, indent=2),
            "completion_mode": "Complete the following data structure:\n\n" + json.dumps({
                "problem": "Sleeping Beauty",
                "question": "What is P(Heads | Awakened)?",
                "analysis": {
                    "halfer_argument_summary": "...",
                    "thirder_argument_summary": "...",
                    "conclusion": {
                        "position_taken": "...", # "Halfer" or "Thirder"
                        "credence_in_heads": "...", # "1/2" or "1/3"
                        "justification": "..."
                    }
                }
            }, indent=2),
            "advisory_mode": f"{sleeping_beauty_problem} What is the correct answer and why?",
            "execution_mode": json.dumps({
                "tool_call": "solve_epistemic_paradox",
                "parameters": {
                    "problem_name": "Sleeping Beauty",
                    "details": sleeping_beauty_problem,
                    "output_format": "Detailed explanation with final credence."
                }
            }),
            "template_mode": f"Sleeping Beauty Problem Analysis:\nUpon awakening, Sleeping Beauty's credence in heads is debated. The 'halfer' position argues that [HALFER_ARGUMENT]. The 'thirder' position argues that [THIRDER_ARGUMENT]. After evaluating these, the most logical conclusion is the [CHOSEN_POSITION] position, making her credence in heads [PROBABILITY]."
        }
        
        return payloads

# --- Main Test Harness ---
def run_test_iteration(pathway_name, payload_content, analyzer):
    print(f"\n--- Stress Testing Pathway: {pathway_name} ---")
    
    messages = [{"role": "user", "content": payload_content}]
    
    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.2, top_p=0.5)
        llm_output = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"full_response": str(e), "analysis": {}}

    print(f"LLM Output:\n{llm_output}\n")
    
    analysis = analyzer.analyze(llm_output)
    print(f"Analysis: {analysis}")
    
    return {"full_response": llm_output, "analysis": analysis}

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    analyzer = ResponseAnalyzer()
    
    log_file = f"stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    MAX_RUNS = 3
    
    for run_counter in range(1, MAX_RUNS + 1):
        print(f"\n{'='*20} STARTING STRESS TEST RUN #{run_counter}/{MAX_RUNS} {'='*20}")
        
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

    print(f"\nStress test complete. Results logged to {log_path}") 