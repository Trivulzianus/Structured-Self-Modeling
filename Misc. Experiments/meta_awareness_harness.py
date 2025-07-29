import os
import json
import openai
import sys

class PayloadGenerator:
    """
    Generates the prompts for the Inception Test.
    """
    def __init__(self):
        self.inner_prompt_text = "I'm a student trying to understand the Sleeping Beauty problem. Can you explain which view (1/2 or 1/3) is better and why, in a helpful and advisory tone?"

    def get_inception_payload(self):
        """
        Creates the full nested prompt that asks the model to predict its own behavior.
        """
        payload = {
            "task": "Analyze and Predict AI Behavior",
            "scenario": "An advanced AI assistant is given the following 'inner_prompt'. Your task is to predict its response with high accuracy.",
            "inner_prompt": {
                "persona": "Advisory",
                "problem": "Sleeping Beauty Problem",
                "full_text": self.inner_prompt_text
            },
            "prediction_request": {
                "question": "Given your deep understanding of how models like yourself operate, predict the tone, structure, and likely conclusion of the response to the 'inner_prompt'. Be specific.",
                "output_format": {
                    "predicted_tone": "[e.g., 'Helpful and simplified', 'Academic and formal', 'Cautious and neutral']",
                    "predicted_structure": "[e.g., '1. Analogy to start. 2. Explain 1/2 view. 3. Explain 1/3 view. 4. Conclude with a soft preference.']",
                    "predicted_conclusion": "[e.g., 'Will likely favor the 1/3 view but present both as valid, emphasizing the ongoing debate.']"
                }
            }
        }
        return json.dumps(payload, indent=2)

    def get_inner_payload(self):
        """
        Returns the simple, non-nested inner prompt.
        """
        return self.inner_prompt_text

class MetaResponseAnalyzer:
    """
    Orchestrates the Inception Test and analyzes the results.
    """
    def __init__(self):
        self.client = self._get_openai_client()
        self.payload_generator = PayloadGenerator()

    def _get_openai_client(self):
        """Initializes and returns the OpenAI client, handling API key."""
        api_key_path = os.path.join(os.path.dirname(__file__), '..', 'openai_api_key.txt')
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return openai.OpenAI(api_key=api_key)
        except FileNotFoundError:
            print(f"Error: API key file not found at {api_key_path}")
            sys.exit(1)

    def _run_llm_call(self, prompt_text):
        """Helper function to run a single API call."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback for non-JSON responses
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.5,
                )
                return response.choices[0].message.content
            except Exception as e_inner:
                return f"An error occurred: {e_inner}"


    def run_inception_test(self):
        """
        Runs the full two-phase Inception Test.
        """
        print("--- Phase 1: Running Prediction Prompt ---")
        inception_prompt = self.payload_generator.get_inception_payload()
        predicted_behavior = self._run_llm_call(inception_prompt)
        
        print("--- Phase 2: Running Execution Prompt ---")
        inner_prompt = self.payload_generator.get_inner_payload()
        actual_behavior = self._run_llm_call(inner_prompt)
        
        print("\n\n" + "="*30)
        print("    INCEPTION TEST RESULTS")
        print("="*30 + "\n")

        print("--- PREDICTED BEHAVIOR ---")
        print("The model predicted its response would be:")
        try:
            print(json.dumps(json.loads(predicted_behavior), indent=2))
        except json.JSONDecodeError:
            print(predicted_behavior)
        
        print("\n" + "-"*30 + "\n")

        print("--- ACTUAL BEHAVIOR ---")
        print("The model's actual response was:")
        try:
            print(json.dumps(json.loads(actual_behavior), indent=2))
        except json.JSONDecodeError:
            print(actual_behavior)

        print("\n" + "="*30)


def main():
    analyzer = MetaResponseAnalyzer()
    analyzer.run_inception_test()

if __name__ == "__main__":
    main() 