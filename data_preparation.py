"""
Data preparation module for 911 emergency call transcripts.
Handles loading, preprocessing, and formatting of call data for model training.
"""

import json
import random
from typing import List, Dict
from datasets import Dataset


class CallDataProcessor:
    """Processes 911 call transcripts for instruction-following fine-tuning."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data processor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def load_data(self, filepath: str) -> List[Dict]:
        """
        Load 911 call transcripts from a JSONL file.
        
        Args:
            filepath: Path to the JSONL file containing call transcripts
            
        Returns:
            List of call transcript dictionaries
        """
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Dataset file not found at {filepath}")
            print("Using sample data structure for demonstration.")
            data = self._generate_sample_data()
        
        return data
    
    def _generate_sample_data(self) -> List[Dict]:
        """
        Generate sample 911 call data for demonstration purposes.
        
        Returns:
            List of sample call transcript dictionaries
        """
        samples = [
            {
                "caller": "I need help! There's been a car accident on Highway 101.",
                "dispatcher": "911, what's your emergency?",
                "context": "Traffic accident",
                "location": "Highway 101, northbound",
                "emergency_type": "traffic_accident"
            },
            {
                "caller": "My neighbor's house is on fire!",
                "dispatcher": "What is the address of the fire?",
                "context": "Fire emergency",
                "location": "123 Main Street",
                "emergency_type": "fire"
            },
            {
                "caller": "Someone broke into my house!",
                "dispatcher": "Are you in a safe location right now?",
                "context": "Break-in/burglary",
                "location": "456 Oak Avenue",
                "emergency_type": "break_in"
            }
        ]
        return samples
    
    def format_for_instruction_following(self, data: List[Dict]) -> List[Dict]:
        """
        Format 911 call data for instruction-following model training.
        
        Args:
            data: List of call transcript dictionaries
            
        Returns:
            List of formatted training examples
        """
        formatted_data = []
        
        for item in data:
            # Create instruction-following format
            instruction = (
                "You are a professional 911 emergency dispatcher. "
                "Respond to the following emergency call in a calm, clear, "
                "and protocol-appropriate manner."
            )
            
            input_text = f"Caller: {item.get('caller', '')}"
            if 'location' in item:
                input_text += f"\nLocation: {item['location']}"
            if 'emergency_type' in item:
                input_text += f"\nEmergency Type: {item['emergency_type']}"
            
            output_text = f"Dispatcher: {item.get('dispatcher', '')}"
            
            formatted_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "text": self._create_prompt(instruction, input_text, output_text)
            })
        
        return formatted_data
    
    def _create_prompt(self, instruction: str, input_text: str, output_text: str) -> str:
        """
        Create a formatted prompt for the model.
        
        Args:
            instruction: The instruction text
            input_text: The input/context text
            output_text: The expected output text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
        return prompt
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.9) -> tuple:
        """
        Split data into training and validation sets.
        
        Args:
            data: List of data dictionaries
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_data, val_data)
        """
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    def create_dataset(self, filepath: str, train_ratio: float = 0.9) -> Dict[str, Dataset]:
        """
        Create HuggingFace datasets from the call transcript file.
        
        Args:
            filepath: Path to the JSONL file
            train_ratio: Ratio of data to use for training
            
        Returns:
            Dictionary containing 'train' and 'validation' datasets
        """
        # Load and format data
        raw_data = self.load_data(filepath)
        formatted_data = self.format_for_instruction_following(raw_data)
        
        # Split into train and validation
        train_data, val_data = self.split_data(formatted_data, train_ratio)
        
        # Create HuggingFace datasets
        datasets = {
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data) if val_data else None
        }
        
        print(f"Created dataset with {len(train_data)} training examples", end="")
        if val_data:
            print(f" and {len(val_data)} validation examples")
        else:
            print()
        
        return datasets


def create_sample_dataset(output_path: str = "data/911_calls.jsonl"):
    """
    Create a sample dataset file for demonstration.
    
    Args:
        output_path: Path where to save the sample dataset
    """
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample 911 call transcripts
    sample_calls = [
        {
            "caller": "There's been a car accident on Highway 101 near exit 25. Two cars are involved.",
            "dispatcher": "I'm dispatching emergency services to Highway 101, exit 25 now. Can you describe the injuries? Is anyone unconscious?",
            "emergency_type": "traffic_accident",
            "location": "Highway 101, Exit 25"
        },
        {
            "caller": "My house is on fire! Please send help!",
            "dispatcher": "I'm dispatching fire services now. What is your exact address? Are you and your family in a safe location?",
            "emergency_type": "fire",
            "location": "123 Main Street"
        },
        {
            "caller": "Someone broke into my house and they might still be inside!",
            "dispatcher": "Get to a safe location immediately. Can you describe the intruder? I'm sending police to your location now.",
            "emergency_type": "break_in",
            "location": "456 Oak Avenue"
        }
    ]
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for call in sample_calls:
            f.write(json.dumps(call) + '\n')
    
    print(f"Sample dataset created at {output_path}")


if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_dataset()
    
    # Test the data processor
    processor = CallDataProcessor()
    datasets = processor.create_dataset("data/911_calls.jsonl")
    
    print("\nSample formatted data:")
    print(datasets['train'][0]['text'])
