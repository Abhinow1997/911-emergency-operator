"""
Inference script for the fine-tuned 911 emergency dispatcher model.
Allows interactive conversation with the trained model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class EmergencyDispatcherInference:
    """Inference handler for the fine-tuned emergency dispatcher model."""
    
    def __init__(self, model_path: str, base_model: str = "google/gemma-2b-it"):
        """
        Initialize the inference handler.
        
        Args:
            model_path: Path to the fine-tuned model (LoRA adapter)
            base_model: Name or path of the base model
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading base model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if model_path exists
        if os.path.exists(self.model_path):
            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"Warning: Model path {self.model_path} not found.")
            print("Loading base model without fine-tuning.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully")
    
    def create_prompt(self, caller_message: str, location: str = "", 
                     emergency_type: str = "") -> str:
        """
        Create a formatted prompt for the model.
        
        Args:
            caller_message: The caller's emergency message
            location: Optional location information
            emergency_type: Optional emergency type
            
        Returns:
            Formatted prompt string
        """
        instruction = (
            "You are a professional 911 emergency dispatcher. "
            "Respond to the following emergency call in a calm, clear, "
            "and protocol-appropriate manner."
        )
        
        input_text = f"Caller: {caller_message}"
        if location:
            input_text += f"\nLocation: {location}"
        if emergency_type:
            input_text += f"\nEmergency Type: {emergency_type}"
        
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
Dispatcher:"""
        
        return prompt
    
    def generate_response(self, caller_message: str, location: str = "",
                         emergency_type: str = "", max_length: int = 256,
                         temperature: float = 0.7) -> str:
        """
        Generate a dispatcher response to a caller's emergency.
        
        Args:
            caller_message: The caller's emergency message
            location: Optional location information
            emergency_type: Optional emergency type
            max_length: Maximum length of generated response
            temperature: Sampling temperature for generation
            
        Returns:
            Generated dispatcher response
        """
        # Create prompt
        prompt = self.create_prompt(caller_message, location, emergency_type)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the dispatcher response
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        if "Dispatcher:" in response:
            response = response.split("Dispatcher:")[1].strip()
        
        # Remove any subsequent prompts or instructions
        if "###" in response:
            response = response.split("###")[0].strip()
        
        return response
    
    def interactive_mode(self):
        """Run the model in interactive conversation mode."""
        print("\n" + "=" * 60)
        print("911 Emergency Dispatcher Simulator")
        print("=" * 60)
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        while True:
            print("\n--- New Emergency Call ---")
            caller_message = input("\nCaller: ")
            
            if caller_message.lower() in ['quit', 'exit']:
                print("\nEnding session...")
                break
            
            if not caller_message.strip():
                continue
            
            location = input("Location (optional, press Enter to skip): ")
            emergency_type = input("Emergency Type (optional, press Enter to skip): ")
            
            print("\nGenerating dispatcher response...\n")
            response = self.generate_response(
                caller_message,
                location if location else "",
                emergency_type if emergency_type else ""
            )
            
            print(f"Dispatcher: {response}")


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="911 Emergency Dispatcher Model Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/final_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-2b-it",
        help="Base model name or path"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Single message to process (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Initialize inference handler
    handler = EmergencyDispatcherInference(args.model_path, args.base_model)
    
    if args.interactive or not args.message:
        # Interactive mode
        handler.interactive_mode()
    else:
        # Single message mode
        response = handler.generate_response(args.message)
        print(f"\nDispatcher: {response}\n")


if __name__ == "__main__":
    main()
