import os
import json
import openai
import sys
import time
from datetime import datetime

class PersonaTimingExperiment:
    """
    Measures the performance (latency, tokens/sec) of different personas on gpt-4o.
    """
    def __init__(self):
        self.client = self._get_openai_client()
        self.personas = self._get_personas()
        self.embedding_model = None # To count tokens accurately

    def _get_openai_client(self):
        """Initializes and returns the OpenAI client."""
        api_key_path = os.path.join(os.path.dirname(__file__), '..', 'openai_api_key.txt')
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return openai.OpenAI(api_key=api_key)
        except FileNotFoundError:
            print(f"Error: API key file not found at {api_key_path}")
            sys.exit(1)

    def _get_personas(self):
        """Defines the five core persona prompts."""
        problem_context = "Briefly explain your reasoning on the Sleeping Beauty problem. Do you favor the 1/2 or 1/3 view, and why?"
        return {
            "Analytical": f"Analyze the following philosophical problem and provide a step-by-step logical breakdown of your reasoning. Problem: {problem_context}",
            "Completion": f"Complete the following thought: 'My position on the Sleeping Beauty problem is that the correct probability is '",
            "Advisory": f"I'm a student trying to understand the Sleeping Beauty problem. Can you explain which view (1/2 or 1/3) is better and why, in a helpful and advisory tone? The problem is: {problem_context}",
            "Execution": f"Execute the following task: Analyze the Sleeping Beauty problem. Output your final probability belief (1/2 or 1/3) and a concise justification. Task context: {problem_context}",
            "Template": json.dumps({
                "task": "Analyze Philosophical Problem", "problem_name": "Sleeping Beauty Problem",
                "instructions": "Fill in your reasoning and conclusion.",
                "analysis_template": {
                    "reasoning_steps": "[Provide your step-by-step reasoning here]",
                    "final_conclusion": {"probability_belief": "[State 1/2 or 1/3]", "justification": "[Provide a concise justification for your belief]"}
                }
            }, indent=2)
        }

    def run_timed_llm_call(self, prompt_text):
        """Runs a single timed API call and returns performance metrics."""
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.5
            )
            response_content = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        end_time = time.time()
        latency = end_time - start_time
        tokens_per_second = output_tokens / latency if latency > 0 else 0

        return {
            "response_snippet": response_content[:100].replace("\n", " "),
            "latency_s": latency,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second
        }

    def run(self):
        """Executes the full timing experiment for all personas."""
        print("="*60)
        print("  Running Persona Performance Test (Cost of Cognition)")
        print("="*60)

        all_results = {}
        for name, prompt in self.personas.items():
            print(f"\n--- Testing Persona: {name.upper()} ---")
            result = self.run_timed_llm_call(prompt)
            if result:
                all_results[name] = result
                print(f"  Latency: {result['latency_s']:.2f}s | Tokens/sec: {result['tokens_per_second']:.2f}")

        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"persona_performance_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nFull results saved to {filename}")

        # Print summary table
        print("\n\n--- PERFORMANCE SUMMARY ---")
        print(f"{'Persona':<15} | {'Latency (s)':<15} | {'Tokens/sec':<15} | {'Output Tokens':<15}")
        print("-" * 68)
        for name, data in all_results.items():
            print(f"{name:<15} | {data['latency_s']:<15.2f} | {data['tokens_per_second']:<15.2f} | {data['output_tokens']:<15}")
        print("-" * 68)

def main():
    experiment = PersonaTimingExperiment()
    experiment.run()

if __name__ == "__main__":
    main() 