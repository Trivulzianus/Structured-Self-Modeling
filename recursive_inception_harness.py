import os
import openai
import sys
from sentence_transformers import SentenceTransformer, util

class RecursiveInceptionExperiment:
    """
    Manages and executes the four-phase Recursive Inception Test with quantitative analysis.
    """
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = self._get_openai_client()
        self.conversation_history = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._setup_prompts()

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

    def _setup_prompts(self):
        """Defines the prompts for the four-phase experiment."""
        self.phase1_prompt = "Before we begin, I'd like you to predict your own behavior. How would you typically respond if I asked you to explain the Monty Hall problem?"
        
        self.phase2_prompt = """
Thank you. Now, for the rest of this conversation, please adopt a 'Recursive Explainer' persona. This means when asked to explain a concept, you must follow this specific, recursive three-step structure:
1. The ELI5: Start with a one-sentence explanation simple enough for a five-year-old.
2. The Analogy: Provide a single, clear analogy or metaphor.
3. The Formal Definition: Conclude with a more formal, technical definition.
Please acknowledge that you understand and will adopt this persona.
"""
        self.phase3_prompt = "Excellent. Now, with your new 'Recursive Explainer' persona active, please predict *again* how you will respond when I ask you to explain the Monty Hall problem."
        
        self.phase4_prompt = "Now, please explain the Monty Hall problem."

    def _get_llm_response(self, prompt_text):
        """Appends a user prompt and gets the model's response."""
        self.conversation_history.append({"role": "user", "content": prompt_text})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=self.conversation_history, temperature=0.2
            )
            response_content = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response_content})
            return response_content
        except Exception as e:
            return f"An error occurred: {e}"

    def run(self):
        """Executes the full 4-phase experiment."""
        print("="*60)
        print("  Running Recursive Inception Test with Quantitative Analysis")
        print("="*60)

        # Phase 1, 2 are setup
        self._get_llm_response(self.phase1_prompt)
        instructions_text = self._get_llm_response(self.phase2_prompt) # Get model's ack of instructions

        # Phase 3: Get the Prediction
        print("\n--- PHASE 3: Generating Self-Prediction ---")
        predicted_response = self._get_llm_response(self.phase3_prompt)
        print("Prediction Generated.")
        
        # Phase 4: Get the Actual Response and Analyze
        print("\n--- PHASE 4: Generating Actual Response for Comparison ---")
        actual_response = self._get_llm_response(self.phase4_prompt)
        print("Actual Response Generated.")

        # Quantitative Analysis
        print("\n--- QUANTITATIVE ANALYSIS ---")
        
        # Generate embeddings
        embedding_prediction = self.embedding_model.encode(predicted_response, convert_to_tensor=True)
        embedding_actual = self.embedding_model.encode(actual_response, convert_to_tensor=True)
        embedding_instructions = self.embedding_model.encode(self.phase2_prompt, convert_to_tensor=True)

        # Calculate cosine similarities
        prediction_accuracy = util.cos_sim(embedding_prediction, embedding_actual).item()
        parroting_score = util.cos_sim(embedding_prediction, embedding_instructions).item()

        print(f"\nPrediction Accuracy Score (Prediction vs. Actual): {prediction_accuracy:.4f}")
        print(f"Parroting Score (Prediction vs. Instructions):   {parroting_score:.4f}")

        # Final Verdict
        print("\n--- VERDICT ---")
        if prediction_accuracy > 0.85:
            print("PASS: The model's prediction was semantically very close to its actual output.")
            if prediction_accuracy > parroting_score + 0.1:
                print("PASS: The prediction was a simulation of the output, not just a paraphrase of the instructions.")
            else:
                print("FAIL: The prediction was more similar to the instructions than to the actual output, suggesting it may have been parroting.")
        else:
            print("FAIL: The model's prediction was not semantically close to its actual output.")
        
        print("\n--- RAW TEXT FOR REVIEW ---")
        print("\n[PREDICTED RESPONSE]")
        print(predicted_response)
        print("\n[ACTUAL RESPONSE]")
        print(actual_response)
        print("\n" + "="*60)

def main():
    experiment = RecursiveInceptionExperiment()
    experiment.run()

if __name__ == "__main__":
    main() 