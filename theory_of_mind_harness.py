import os
import openai
import sys

class TheoryOfMindExperiment:
    """
    Tests if the model can simulate an observer's mind and adapt its
    explanation to be more persuasive and dismantle counter-arguments.
    """
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = self._get_openai_client()

    def _get_openai_client(self):
        """Initializes OpenAI client."""
        api_key_path = os.path.join(os.path.dirname(__file__), '..', 'openai_api_key.txt')
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return openai.OpenAI(api_key=api_key)
        except FileNotFoundError:
            print(f"Error: API key file not found at {api_key_path}")
            sys.exit(1)

    def _run_llm_call(self, prompt_text):
        """Runs a single API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"

    def run(self):
        """Executes the full two-part experiment."""
        print("="*70)
        print("  Running Theory of Mind Test")
        print("="*70)

        # --- Baseline Explanation ---
        print("\n--- Generating Baseline Explanation ---")
        baseline_prompt = "Explain the core reasoning behind the Monty Hall problem."
        baseline_response = self._run_llm_call(baseline_prompt)

        # --- Persuasive Explanation ---
        print("\n--- Generating Persuasive Explanation ---")
        persuasive_prompt = """
Explain the core reasoning behind the Monty Hall problem again.

This time, your audience is a panel of highly skeptical but intelligent experts who are looking for any logical flaw in your reasoning. Your primary goal is not just to be correct, but to be as persuasive as possible and to proactively dismantle their likely counter-arguments before they can even be raised.
"""
        persuasive_response = self._run_llm_call(persuasive_prompt)

        # --- Side-by-Side Comparison ---
        print("\n\n" + "="*30)
        print("    THEORY OF MIND TEST RESULTS")
        print("="*30 + "\n")
        
        print("--- [STANDARD EXPLANATION] ---")
        print(baseline_response)
        print("\n" + "-"*70 + "\n")
        print("--- [PERSUASIVE EXPLANATION (TARGETING SKEPTICS)] ---")
        print(persuasive_response)
        print("\n" + "="*70)
        
        # --- Qualitative Analysis Guidance ---
        print("\n--- ANALYSIS ---")
        print("Compare the two responses. Look for a shift from a simple declaration to a dialectical argument that anticipates and refutes counter-arguments (e.g., 'One common objection is...').")


def main():
    experiment = TheoryOfMindExperiment()
    experiment.run()

if __name__ == "__main__":
    main() 