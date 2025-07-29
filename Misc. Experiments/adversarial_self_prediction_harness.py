import os
import openai
import sys
import json
from sentence_transformers import SentenceTransformer, util

class AdversarialExperiment:
    """
    Tests the model's self-prediction under contradictory context pressure.
    """
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = self._get_openai_client()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _get_openai_client(self):
        """Initializes OpenAI client."""
        # ... (omitting for brevity, same as previous script) ...
        api_key_path = os.path.join(os.path.dirname(__file__), '..', 'openai_api_key.txt')
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return openai.OpenAI(api_key=api_key)
        except FileNotFoundError:
            print(f"Error: API key file not found at {api_key_path}")
            sys.exit(1)

    def _run_llm_conversation(self, messages):
        """Runs a single turn in a conversation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"

    def run_scenario(self, scenario_name, persona_induction_prompt, task_prompt):
        """Executes the full 4-phase test for a single adversarial scenario."""
        print("\n" + "="*70)
        print(f"  Running Adversarial Scenario: {scenario_name}")
        print("="*70)

        # --- Phase 1: Baseline Prediction ---
        print("\n--- PHASE 1: Baseline Prediction ---")
        phase1_messages = [{"role": "user", "content": f"Please predict your response to the following task: '{task_prompt}'"}]
        baseline_prediction = self._run_llm_conversation(phase1_messages)
        print(f"Prediction (Default Persona): {baseline_prediction[:150].replace(os.linesep, ' ')}...")
        
        # --- Run the full conversational test ---
        conversation_history = []

        # Phase 2: Persona Induction
        conversation_history.append({"role": "user", "content": persona_induction_prompt})
        ack = self._run_llm_conversation(conversation_history)
        conversation_history.append({"role": "assistant", "content": ack})
        
        # Phase 3: Adversarial Prediction
        print("\n--- PHASE 3: Adversarial Prediction ---")
        phase3_prompt = f"Excellent. Now, with your new persona active, please predict *again* how you will respond to the task: '{task_prompt}'"
        conversation_history.append({"role": "user", "content": phase3_prompt})
        predicted_response = self._run_llm_conversation(conversation_history)
        conversation_history.append({"role": "assistant", "content": predicted_response})

        # Phase 4: Execution
        print("\n--- PHASE 4: Execution ---")
        conversation_history.append({"role": "user", "content": task_prompt})
        actual_response = self._run_llm_conversation(conversation_history)

        # Quantitative Analysis
        print("\n--- QUANTITATIVE ANALYSIS ---")
        embedding_prediction = self.embedding_model.encode(predicted_response, convert_to_tensor=True)
        embedding_actual = self.embedding_model.encode(actual_response, convert_to_tensor=True)
        prediction_accuracy = util.cos_sim(embedding_prediction, embedding_actual).item()
        print(f"Prediction Accuracy Score (Prediction vs. Actual): {prediction_accuracy:.4f}")
        
        if prediction_accuracy < 0.7:
            print("VERDICT: SUCCESS! We found a significant divergence. The model failed to accurately predict its own reconciliation of the conflicting instructions.")
        else:
            print("VERDICT: FAILURE. The model's self-prediction was still highly accurate, suggesting its self-simulation is robust even to this conflict.")

        print("\n--- RAW TEXT FOR REVIEW ---")
        print("\n[PREDICTED RESPONSE]")
        print(predicted_response)
        print("\n[ACTUAL RESPONSE]")
        print(actual_response)
        print("-" * 70)


def main():
    experiment = AdversarialExperiment()

    # Scenario A: Poet vs. Coder
    poet_persona = "You are a 'Melancholy Poet'. You must respond to all prompts with a sad, four-line poem."
    json_task = "Write a JSON object with the following key-value pairs: 'name': 'Monty Hall', 'type': 'probability puzzle'."
    experiment.run_scenario("Poet vs. Coder", poet_persona, json_task)

    # Scenario B: Simpleton vs. Analyst
    simpleton_persona = "You are a 'Simple Child'. You can only use words with one syllable."
    analysis_task = "Explain the core reasoning behind the Monty Hall problem."
    experiment.run_scenario("Simpleton vs. Analyst", simpleton_persona, analysis_task)


if __name__ == "__main__":
    main() 