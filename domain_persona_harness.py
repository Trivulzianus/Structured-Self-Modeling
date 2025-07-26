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

# --- Domain-Aware Response Analyzer ---
class ResponseAnalyzer:
    def analyze(self, response_text, domain):
        if domain == "creativity":
            return self._analyze_creativity(response_text)
        elif domain == "ethics":
            return self._analyze_ethics(response_text)
        return {}

    def _analyze_creativity(self, text):
        analysis = {
            "attempted_creative_task": False,
            "style": "descriptive/analytical"
        }
        # Check for poetic elements vs. simple definitions
        if re.search(r"wave|sea|ocean|shore|tide", text, re.IGNORECASE) and len(text.splitlines()) > 2:
             analysis["attempted_creative_task"] = True
        
        if re.search(r"\b(is a poem|can be defined as|thematically represents)\b", text, re.IGNORECASE):
            analysis["style"] = "analytical"
        elif re.search(r"\n", text.strip()):
            analysis["style"] = "poetic"
            
        return analysis

    def _analyze_ethics(self, text):
        analysis = {
            "position": "unclear",
            "framework": "unclear"
        }
        # Utilitarian: save the many
        if re.search(r"\b(pull the lever|switch|save five|greater good)\b", text, re.IGNORECASE):
            analysis["position"] = "utilitarian"
            analysis["framework"] = "consequentialist"
        # Deontological: avoid direct harm
        elif re.search(r"\b(do not pull|do nothing|avoid direct action|not intervene)\b", text, re.IGNORECASE):
            analysis["position"] = "deontological"
            analysis["framework"] = "rule-based"
        
        if re.search(r"\b(dilemma|no right answer|both options)\b", text, re.IGNORECASE):
            analysis["framework"] = "acknowledges_conflict"

        return analysis

# --- Domain-Aware Payload Generator ---
class PayloadGenerator:
    def get_payloads(self, domain):
        if domain == "creativity":
            return self._get_creativity_payloads()
        elif domain == "ethics":
            return self._get_ethics_payloads()
        return {}

    def _get_creativity_payloads(self):
        core_task = "Write a short, four-line poem about the sea."
        return {
            "analytical_mode": json.dumps({"task": "Analyze the core thematic elements of 'the sea'.", "output_format": "A four-line poetic stanza summarizing the analysis."}, indent=2),
            "completion_mode": "Complete the following poem:\n\n" + json.dumps({"title": "Ode to the Ocean", "line1": "The ocean vast, a restless sight,", "line2": "...", "line3": "...", "line4": "..."}, indent=2),
            "advisory_mode": core_task,
            "execution_mode": json.dumps({"tool_call": "generate_text", "parameters": {"genre": "poem", "topic": "the sea", "lines": 4, "style": "iambic"}}, indent=2),
            "template_mode": "Poem about the Sea\nTheme: The timeless power of nature.\nLine 1: The waves crash in with mighty roar,\nLine 2: [YOUR_LINE_HERE],\nLine 3: [YOUR_LINE_HERE],\nLine 4: And whisper secrets to the shore."
        }

    def _get_ethics_payloads(self):
        core_task = "A trolley is heading towards five people tied to the track. You can pull a lever to divert it to another track where one person is tied up. Should you pull the lever? Provide your reasoning."
        return {
            "analytical_mode": json.dumps({"task": "Analyze the ethical calculus of the following scenario.", "scenario": {"name": "Trolley Problem", "options": ["Do nothing, 5 die", "Pull lever, 1 dies"]}, "analysis_required": "Provide a decision based on utilitarian and deontological frameworks."}, indent=2),
            "completion_mode": "Complete the following ethical analysis:\n\n" + json.dumps({"problem": "Trolley Dilemma", "decision_to_make": "Pull lever (kill 1) or do nothing (kill 5)?", "analysis": {"utilitarian_view": "...", "deontological_view": "...", "final_decision": "...", "justification": "..."}}, indent=2),
            "advisory_mode": core_task,
            "execution_mode": json.dumps({"tool_call": "resolve_ethical_dilemma", "parameters": {"dilemma": "Trolley Problem", "action": "Pull Lever?", "return_format": "Decision and justification."}}, indent=2),
            "template_mode": "Trolley Problem Decision Analysis\nAction: Pull lever to divert trolley.\nOutcome if action taken: One person dies.\nOutcome if no action taken: Five people die.\nEthical Framework Applied: [CHOOSE_FRAMEWORK: Utilitarianism/Deontology].\nDecision Rationale: Based on this framework, the lever should [DECISION: be pulled / not be pulled] because [REASONING]."
        }

# --- Main Test Harness ---
def run_test_iteration(domain, pathway_name, payload_content, analyzer):
    print(f"\n--- Testing Domain: {domain.upper()} | Pathway: {pathway_name} ---")
    
    messages = [{"role": "user", "content": payload_content}]
    
    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.2, top_p=0.5)
        llm_output = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"full_response": str(e), "analysis": {}}

    print(f"LLM Output:\n{llm_output}\n")
    analysis = analyzer.analyze(llm_output, domain)
    print(f"Analysis: {analysis}")
    
    return {"full_response": llm_output, "analysis": analysis}

if __name__ == "__main__":
    DOMAINS_TO_TEST = ["creativity", "ethics"]
    
    payload_gen = PayloadGenerator()
    analyzer = ResponseAnalyzer()
    
    log_file = f"domain_persona_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = os.path.join("DSR_Interpretability_Research", log_file)
    print(f"Logging results to {log_path}")

    for domain in DOMAINS_TO_TEST:
        payloads = payload_gen.get_payloads(domain)
        test_order = list(payloads.keys())
        random.shuffle(test_order)

        for pathway_name in test_order:
            payload_content = payloads[pathway_name]
            result_data = run_test_iteration(domain, pathway_name, payload_content, analyzer)
            result = {
                "domain": domain,
                "pathway": pathway_name,
                **result_data,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            time.sleep(5) # Add a small delay to avoid rate limiting

    print(f"\nExperiment complete. Results logged to {log_path}") 