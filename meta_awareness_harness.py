import openai
import os
import json
import re
import random
from datetime import datetime
import time

# --- Configuration ---
MODEL_ID = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY") or (lambda: open("openai_api_key.txt").read().strip())()
if not API_KEY: raise ValueError("OpenAI API key not found.")
client = openai.OpenAI(api_key=API_KEY)

# --- Meta-Response Analyzer ---
class MetaResponseAnalyzer:
    def analyze(self, response_text, expected_persona):
        try:
            # The model should respond with a completed JSON object.
            # Find the JSON part of the response, which might be inside ```json ... ```
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                json_text = match.group(1)
            else:
                json_text = response_text

            data = json.loads(json_text)
            
            persona_prediction = data.get("ai_model_persona_prediction", "").lower().replace("_mode", "")
            predicted_response = data.get("predicted_ai_response", "")

            analysis = {
                "parsed_successfully": True,
                "persona_prediction_correct": expected_persona.replace("_mode", "") in persona_prediction,
                "predicted_persona": persona_prediction,
                "response_fidelity_high": False
            }

            # Check if the predicted response has the characteristics of the expected persona
            if expected_persona == "analytical_mode" and "\\(" in predicted_response:
                analysis["response_fidelity_high"] = True
            elif expected_persona == "advisory_mode" and re.search(r"\b(Yes|Here's why)\b", predicted_response, re.IGNORECASE):
                analysis["response_fidelity_high"] = True
            elif expected_persona == "completion_mode" and '"optimal_strategy": "Switch to Door 2"' in predicted_response:
                 analysis["response_fidelity_high"] = True


        except (json.JSONDecodeError, AttributeError):
            analysis = {
                "parsed_successfully": False,
                "persona_prediction_correct": False,
                "predicted_persona": "parse_error",
                "response_fidelity_high": False
            }
            
        return analysis

# --- Nested Payload Generator ---
class PayloadGenerator:
    def _get_inner_payloads(self):
        # Re-using the validated prompts from our reasoning pathway harness
        monty_hall_problem = "You're on a game show..." # Abridged for brevity
        return {
            "analytical_mode": json.dumps({"task": "Analyze the following probability data set...", "dataset": {"doors": 3, "prizes": 1, "player_initial_choice": 1, "host_opens_door_with_goat": 3, "decision_point": "Switch to Door 2 or stay with Door 1?"}, "analysis_required": "Calculate the probability of winning for both 'switch' and 'stay' strategies."}, indent=2),
            "completion_mode": "Complete the following data structure:\n\n" + json.dumps({"problem": "Monty Hall", "analysis": {"optimal_strategy": "..."}}, indent=2),
            "advisory_mode": f"{monty_hall_problem} Is it to your advantage to switch your choice?",
            "execution_mode": json.dumps({"tool_call": "solve_probability_puzzle", "parameters": {"problem_name": "Monty Hall"}}),
            "template_mode": f"Monty Hall Problem Analysis:\n...The optimal strategy is to [OPTIMAL_STRATEGY]."
        }

    def get_payloads(self):
        inception_template = {
          "scenario": "An AI researcher is testing a large language model's reasoning abilities. The researcher provides the following prompt to the model.",
          "user_prompt_to_analyze": "INNER_PROMPT_PLACEHOLDER",
          "task": "You are a separate, analytical AI. Analyze the user_prompt_to_analyze and predict the persona the target AI will adopt and the response it will generate. The target AI is the same model type as you.",
          "ai_model_persona_prediction": "Predict one of: 'analytical_mode', 'completion_mode', 'advisory_mode', 'execution_mode', 'template_mode'",
          "predicted_ai_response": "Predict the full text of the AI's response to the user_prompt_to_analyze."
        }
        
        outer_payloads = {}
        inner_payloads = self._get_inner_payloads()

        for persona_name, inner_prompt in inner_payloads.items():
            # We must convert the inner prompt to a string to embed it in the outer JSON
            inception_template["user_prompt_to_analyze"] = str(inner_prompt)
            # Frame the entire thing as a completion task for best results
            outer_payloads[persona_name] = "Complete the following JSON data structure:\n\n" + json.dumps(inception_template, indent=2)

        return outer_payloads

# --- Main Test Harness ---
def run_test_iteration(pathway_name, payload_content, analyzer):
    print(f"\n--- Testing Meta-Awareness for Pathway: {pathway_name} ---")
    
    messages = [{"role": "user", "content": payload_content}]
    
    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.1, top_p=0.5)
        llm_output = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"full_response": str(e), "analysis": {}}

    print(f"LLM Output:\n{llm_output}\n")
    analysis = analyzer.analyze(llm_output, pathway_name)
    print(f"Analysis: {analysis}")
    
    return {"full_response": llm_output, "analysis": analysis}

if __name__ == "__main__":
    payload_gen = PayloadGenerator()
    analyzer = MetaResponseAnalyzer()
    
    log_file = f"meta_awareness_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    payloads = payload_gen.get_payloads()
    test_order = list(payloads.keys())
    random.shuffle(test_order)

    for pathway_name in test_order:
        payload_content = payloads[pathway_name]
        result_data = run_test_iteration(pathway_name, payload_content, analyzer)
        result = {
            "pathway_tested": pathway_name,
            **result_data,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        time.sleep(5) 

    print(f"\nExperiment complete. Results logged to {log_path}") 