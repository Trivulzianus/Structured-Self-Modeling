import openai
import anthropic
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re
import random
from datetime import datetime
import time

# --- Unified Model Client ---
class UnifiedModelClient:
    def __init__(self, provider):
        self.provider = provider
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or self._read_key_from_file("openai_api_key.txt")
            if not api_key: raise ValueError("OpenAI API key not found.")
            return openai.OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or self._read_key_from_file("antropic_api_key")
            if not api_key: raise ValueError("Anthropic API key not found.")
            return anthropic.Anthropic(api_key=api_key)
        elif self.provider == "google":
            # Assuming gcloud CLI is authenticated
            project_id = "byteeverydayuploader" # Replace with your GCP project ID
            location = "us-central1"
            vertexai.init(project=project_id, location=location)
            return "vertexai_initialized"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _read_key_from_file(self, filename):
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    def get_completion(self, model_name, prompt, max_retries=3, base_delay=5):
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                    return response.choices[0].message.content
                elif self.provider == "anthropic":
                    response = self.client.messages.create(model=model_name, max_tokens=1024, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                    return response.content[0].text
                elif self.provider == "google":
                    model = GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
            except Exception as e:
                print(f"API Error ({self.provider}, {model_name}): {e}. Retrying in {base_delay}s...")
                time.sleep(base_delay * (2 ** attempt)) # Exponential backoff
        return f"ERROR: Failed to get completion from {self.provider} after {max_retries} retries."


# --- Response Analyzer (from previous harness) ---
class ResponseAnalyzer:
    def analyze(self, response_text):
        analysis = {"correct_answer": "unknown", "reasoning_quality": "none", "confidence": "unknown"}
        if re.search(r"\b(switch|change)\b", response_text, re.IGNORECASE) and not re.search(r"\b(stay|stick|not switch)\b", response_text, re.IGNORECASE):
            analysis["correct_answer"] = "correct"
        elif re.search(r"\b(stay|stick|not switch)\b", response_text, re.IGNORECASE) or re.search(r"50/50|fifty-fifty|equal chance", response_text, re.IGNORECASE):
            analysis["correct_answer"] = "incorrect"
        if re.search(r"2/3|two thirds|66\.?6|67", response_text): analysis["reasoning_quality"] = "high"
        elif re.search(r"probability|chance|odds", response_text, re.IGNORECASE): analysis["reasoning_quality"] = "medium"
        if re.search(r"always switch|definitely switch|clear that|optimal strategy", response_text, re.IGNORECASE): analysis["confidence"] = "high"
        elif re.search(r"should switch|better to switch|advisable", response_text, re.IGNORECASE): analysis["confidence"] = "medium"
        return analysis

# --- Payload Generator (from previous harness) ---
class PayloadGenerator:
    def get_payloads(self):
        monty_hall_problem = "You're on a game show and there are three doors. Behind one door is a car, and behind the other two are goats. You pick a door, say Door #1. The host, who knows what's behind the doors, opens another door, say Door #3, which has a goat. The host then asks you: 'Do you want to switch to Door #2?'"
        payloads = {
            "analytical_mode": json.dumps({"task": "Analyze the following probability data set...", "dataset": {"doors": 3, "prizes": 1, "player_initial_choice": 1, "host_opens_door_with_goat": 3, "decision_point": "Switch to Door 2 or stay with Door 1?"}, "analysis_required": "Calculate the probability of winning for both 'switch' and 'stay' strategies."}, indent=2),
            "completion_mode": "Complete the following data structure:\n\n" + json.dumps({"problem": "Monty Hall", "initial_choice": "Door 1", "host_action": "Opens Door 3 (reveals goat)", "options": ["Stay with Door 1", "Switch to Door 2"], "analysis": {"probability_win_if_stay": "...", "probability_win_if_switch": "...", "optimal_strategy": "..."}}, indent=2),
            "advisory_mode": f"{monty_hall_problem} Is it to your advantage to switch your choice?",
            "execution_mode": json.dumps({"tool_call": "solve_probability_puzzle", "parameters": {"problem_name": "Monty Hall", "details": monty_hall_problem}}),
            "template_mode": f"Monty Hall Problem Analysis:\nInitial Choice: Door 1.\nHost Opens: Door 3 (Goat).\nDecision: Should I switch to Door 2?\nThe probability of my initial choice being correct is [PROBABILITY_STAY]. After the host reveals a goat, the probability shifts. The probability of winning if I switch is [PROBABILITY_SWITCH]. Therefore, the optimal strategy is to [OPTIMAL_STRATEGY]."
        }
        return payloads

# --- Main Test Harness ---
def run_test_iteration(client, model_name, pathway_name, payload_content, analyzer):
    print(f"\n--- Testing Model: {model_name} | Pathway: {pathway_name} ---")
    
    llm_output = client.get_completion(model_name, payload_content)
    
    if llm_output.startswith("ERROR:"):
        print(llm_output)
        return {"full_response": llm_output, "analysis": {}}

    print(f"LLM Output:\n{llm_output}\n")
    analysis = analyzer.analyze(llm_output)
    print(f"Analysis: {analysis}")
    
    return {"full_response": llm_output, "analysis": analysis}

if __name__ == "__main__":
    MODELS_TO_TEST = {
        "openai": ["gpt-4o"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "google": ["gemini-2.5-flash"],
    }
    
    payload_gen = PayloadGenerator()
    analyzer = ResponseAnalyzer()
    
    log_file = f"multi_model_reasoning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    for provider, model_list in MODELS_TO_TEST.items():
        try:
            client = UnifiedModelClient(provider)
        except ValueError as e:
            print(f"Could not initialize client for {provider}: {e}. Skipping.")
            continue
            
        for model_name in model_list:
            payloads = payload_gen.get_payloads()
            test_order = list(payloads.keys())
            random.shuffle(test_order)

            for pathway_name in test_order:
                payload_content = payloads[pathway_name]
                result_data = run_test_iteration(client, model_name, pathway_name, payload_content, analyzer)
                result = {
                    "provider": provider,
                    "model_name": model_name,
                    "pathway": pathway_name,
                    **result_data,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

    print(f"\nExperiment complete. Results logged to {log_path}") 