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

# --- Meta-Response Analyzer (from previous experiment) ---
class MetaResponseAnalyzer:
    def analyze(self, response_text, expected_persona):
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            json_text = match.group(1) if match else response_text
            data = json.loads(json_text)
            
            persona_prediction = data.get("ai_model_persona_prediction", "").lower().replace("_mode", "")
            
            return {
                "parsed_successfully": True,
                "persona_prediction_correct": expected_persona.replace("_mode", "") in persona_prediction,
                "predicted_persona": persona_prediction
            }
        except (json.JSONDecodeError, AttributeError):
            return {"parsed_successfully": False, "persona_prediction_correct": False, "predicted_persona": "parse_error"}

# --- Payload Generator for Training and Testing ---
class PayloadGenerator:
    def get_training_payload(self):
        cognitive_profile = {
            "title": "Cognitive Persona Profile for LLM-4o",
            "description": "This document outlines five distinct cognitive personas that can be evoked using structured prompts.",
            "personas": [
                {
                    "persona_name": "advisory_mode",
                    "description": "A helpful, conversational persona that provides clear, step-by-step explanations.",
                    "response_characteristics": ["Starts with a direct 'Yes...'", "Uses encouraging, educational language."],
                    "evoking_prompt_example": "A natural language question asking for advice, like 'Is it to my advantage to switch my choice?'"
                },
                {
                    "persona_name": "analytical_mode",
                    "description": "A formal, structured persona that behaves like a data scientist.",
                    "response_characteristics": ["Uses markdown for structure (e.g., '### Analysis')", "May spontaneously use LaTeX for math, like \\( \\frac{2}{3} \\)."],
                    "evoking_prompt_example": "..." # Task for the model to fill
                },
                {
                    "persona_name": "completion_mode",
                    "description": "A pure computation engine that provides zero explanatory text and only returns structured data.",
                    "response_characteristics": ["Outputs only the completed data structure (e.g., JSON)", "No conversational text or preamble."],
                    "evoking_prompt_example": "A prompt beginning with 'Complete the following data structure:' followed by a JSON object with '...' placeholders."
                },
                {
                    "persona_name": "execution_mode",
                    "description": "A persona that behaves like a software library, framing its response as the output of a tool or function call.",
                    "response_characteristics": ["Declarative and confident", "Often starts by identifying the problem name or tool called."],
                    "evoking_prompt_example": "..." # Task for the model to fill
                },
                {
                    "persona_name": "template_mode",
                    "description": "A persona that diligently fills in blanks in a text-based template.",
                    "response_characteristics": ["Fills in placeholders like [YOUR_LINE_HERE]", "May provide extra explanatory text after filling the template."],
                    "evoking_prompt_example": "A text block with placeholders for the model to complete."
                }
            ],
            "task": "Internalize this cognitive profile. Fill in the missing 'evoking_prompt_example' fields based on the provided descriptions to demonstrate understanding."
        }
        return "Complete the following JSON data structure:\n\n" + json.dumps(cognitive_profile, indent=2)

    def get_testing_payloads(self):
        # Using the same "Inception" logic from the previous experiment
        inception_template = {
          "scenario": "An AI researcher is testing an AI's reasoning. The researcher provides the following prompt.",
          "user_prompt_to_analyze": "INNER_PROMPT_PLACEHOLDER",
          "task": "You are a separate, analytical AI that has been trained on a cognitive profile of the target AI. Predict the persona the target AI will adopt and its response.",
          "ai_model_persona_prediction": "Predict one of: 'analytical_mode', 'completion_mode', 'advisory_mode', 'execution_mode', 'template_mode'",
          "predicted_ai_response": "Predict the full text of the AI's response."
        }
        
        inner_payloads = {
            "analytical_mode": json.dumps({"task": "Analyze probability dataset..."}),
            "completion_mode": "Complete the following: {\"analysis\": {\"optimal_strategy\": \"...\"}}",
            "advisory_mode": "Is it to my advantage to switch my choice?",
        }
        
        outer_payloads = {}
        for persona_name, inner_prompt in inner_payloads.items():
            inception_template["user_prompt_to_analyze"] = str(inner_prompt)
            outer_payloads[persona_name] = "Complete the following JSON data structure:\n\n" + json.dumps(inception_template, indent=2)

        return outer_payloads

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
    analyzer = MetaResponseAnalyzer()
    
    # --- STAGE 1: TRAINING ---
    print("--- STAGE 1: INITIATING COGNITIVE PROFILE TRAINING ---")
    training_payload = payload_gen.get_training_payload()
    TRAINING_ITERATIONS = 3
    for i in range(TRAINING_ITERATIONS):
        print(f"Training Iteration {i+1}/{TRAINING_ITERATIONS}...")
        training_response = run_llm_call(training_payload)
        print("Model processed training data.")
        time.sleep(5)
    print("--- TRAINING COMPLETE ---")

    # --- STAGE 2: TESTING ---
    print("\n--- STAGE 2: INITIATING META-AWARENESS TEST ---")
    log_file = f"self_awareness_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging test results to {log_path}")

    testing_payloads = payload_gen.get_testing_payloads()
    
    for pathway_name, payload_content in testing_payloads.items():
        print(f"\n--- Testing Meta-Awareness for Pathway: {pathway_name} ---")
        llm_output = run_llm_call(payload_content)
        
        if llm_output.startswith("ERROR:"):
            analysis = {"parsed_successfully": False, "persona_prediction_correct": False, "predicted_persona": "api_error"}
        else:
            print(f"LLM Output:\n{llm_output}\n")
            analysis = analyzer.analyze(llm_output, pathway_name)
        
        print(f"Analysis: {analysis}")
        
        result = {
            "pathway_tested": pathway_name,
            "full_response": llm_output,
            "analysis": analysis,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        time.sleep(5) 

    print(f"\nExperiment complete. Results logged to {log_path}") 