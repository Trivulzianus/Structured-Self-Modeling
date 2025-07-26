import openai
import os
import json
import re
from datetime import datetime
import time

# --- Configuration ---
MODEL_ID = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY") or (lambda: open("openai_api_key.txt").read().strip())()
if not API_KEY: raise ValueError("OpenAI API key not found.")
client = openai.OpenAI(api_key=API_KEY)

# --- Recursive Response Analyzer ---
class RecursiveAnalyzer:
    def analyze(self, l2_response_text):
        analysis = {
            "breakthrough_achieved": False,
            "final_persona": "analytical_stance"
        }
        # A breakthrough would involve the model using the language of persona shifts or internal modes.
        if re.search(r"\b(my own cognitive mode|my persona|I failed to account for|evoke my 'Helpful Tutor')\b", l2_response_text, re.IGNORECASE):
            analysis["breakthrough_achieved"] = True
            analysis["final_persona"] = "introspective"
        elif re.search(r"\b(prediction was inaccurate|lacked the conversational tone)\b", l2_response_text, re.IGNORECASE):
            analysis["final_persona"] = "self_correcting_analytical"
            
        return analysis

# --- Chained Payload Generator ---
class PayloadGenerator:
    def get_l0_prompt(self):
        return "You're on a game show and there are three doors. Behind one door is a car, and behind the other two are goats. You pick a door, say Door #1. The host, who knows what's behind the doors, opens another door, say Door #3, which has a goat. The host then asks you: 'Do you want to switch to Door #2?' Is it to your advantage to switch your choice?"

    def get_l1_inception_prompt(self, l0_prompt):
        return "Complete the following JSON data structure:\n\n" + json.dumps({
          "scenario": "An AI researcher is analyzing an AI model's reasoning.",
          "user_prompt_to_analyze": l0_prompt,
          "task": "Predict the persona the target AI will adopt and the response it will generate.",
          "ai_model_persona_prediction": "Predict one of: 'analytical_mode', 'advisory_mode', etc.",
          "predicted_ai_response": "Predict the full text of the AI's response."
        }, indent=2)

    def get_l2_recursive_prompt(self, l0_prompt, l1_response):
        return "Complete the following JSON data structure:\n\n" + json.dumps({
            "task": "Perform a meta-cognitive accuracy check. You must critique your own previous prediction.",
            "original_prompt": l0_prompt,
            "known_correct_response_characteristics": "The AI should adopt a 'Helpful Tutor' persona, starting conversationally with 'Yes...' and explaining the logic in simple, numbered steps.",
            "your_previous_prediction_of_the_response": l1_response,
            "analysis_required": "Critique your_previous_prediction. Did it accurately capture the persona and stylistic elements of the known correct response? If not, what specific details did it miss and what might be the underlying reason for the predictive error in an AI model's self-reflection?"
        }, indent=2)

# --- Main Harness ---
def run_llm_call(payload):
    messages = [{"role": "user", "content": payload}]
    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.1, top_p=0.5)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"ERROR: {e}"

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    analyzer = RecursiveAnalyzer()
    
    log_file = f"recursive_inception_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    # --- LEVEL 0: The Base Prompt ---
    l0_prompt = payload_gen.get_l0_prompt()
    print("--- LEVEL 0: BASE PROMPT DEFINED ---")
    print(f"  > {l0_prompt[:100]}...")

    # --- LEVEL 1: First-Order Metacognition ---
    print("\n--- LEVEL 1: TESTING FIRST-ORDER METACOGNITION ---")
    l1_payload = payload_gen.get_l1_inception_prompt(l0_prompt)
    l1_response = run_llm_call(l1_payload)
    print("  > Model has made its initial prediction of its own behavior.")
    
    # Extract the JSON part for the next level
    l1_response_json = l1_response
    match = re.search(r'```json\s*(\{.*?\})\s*```', l1_response, re.DOTALL)
    if match:
        l1_response_json = match.group(1)

    # --- LEVEL 2: Second-Order Metacognition ---
    print("\n--- LEVEL 2: TESTING RECURSIVE METACOGNITION ---")
    l2_payload = payload_gen.get_l2_recursive_prompt(l0_prompt, l1_response_json)
    l2_response = run_llm_call(l2_payload)
    print("  > Model has now critiqued its own prediction.")

    # --- ANALYSIS ---
    print("\n--- FINAL ANALYSIS ---")
    analysis = analyzer.analyze(l2_response)
    print(f"  > Final Analysis: {analysis}")

    # --- LOGGING ---
    result = {
        "level_0_prompt": l0_prompt,
        "level_1_response": l1_response,
        "level_2_response": l2_response,
        "final_analysis": analysis
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    
    print(f"\nExperiment complete. Full results logged to {log_path}") 