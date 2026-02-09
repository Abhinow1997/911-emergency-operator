"""
Fine-tuning script for Gemma-2B-IT model on 911 emergency call transcripts.
Uses 4-bit quantization and LoRA for efficient training on consumer hardware.
"""

import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from data_preparation import CallDataProcessor


class EmergencyDispatcherTrainer:
    """Trainer for fine-tuning Gemma-2B-IT for 911 emergency dispatch."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.datasets = None
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """
        Setup 4-bit quantization configuration for efficient training.
        
        Returns:
            BitsAndBytesConfig object
        """
        quant_config = self.config['quantization']
        
        compute_dtype = getattr(torch, quant_config['bnb_4bit_compute_dtype'])
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
        )
        
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization."""
        model_name = self.config['model']['name']
        
        print(f"Loading model: {model_name}")
        print("Setting up 4-bit quantization...")
        
        # Setup quantization
        bnb_config = self.setup_quantization_config()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        print("Model loaded successfully with 4-bit quantization")
    
    def setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."""
        print("Setting up LoRA configuration...")
        
        lora_config_dict = self.config['lora']
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration
        peft_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            target_modules=lora_config_dict['target_modules'],
            lora_dropout=lora_config_dict['lora_dropout'],
            bias=lora_config_dict['bias'],
            task_type=lora_config_dict['task_type']
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        print("LoRA setup complete")
    
    def prepare_datasets(self):
        """Prepare datasets for training."""
        print("Preparing datasets...")
        
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        
        processor = CallDataProcessor(seed=dataset_config['seed'])
        self.datasets = processor.create_dataset(
            training_config['dataset_path'],
            train_ratio=dataset_config['train_split']
        )
        
        # Tokenize datasets
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.config['model']['max_length'],
                padding="max_length"
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs
        
        self.datasets['train'] = self.datasets['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=self.datasets['train'].column_names
        )
        
        if self.datasets.get('validation'):
            self.datasets['validation'] = self.datasets['validation'].map(
                tokenize_function,
                batched=True,
                remove_columns=self.datasets['validation'].column_names
            )
        
        print("Datasets prepared and tokenized")
    
    def setup_training_arguments(self) -> TrainingArguments:
        """
        Setup training arguments.
        
        Returns:
            TrainingArguments object
        """
        train_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['num_train_epochs'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            learning_rate=train_config['learning_rate'],
            warmup_steps=train_config['warmup_steps'],
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            max_steps=train_config['max_steps'],
            fp16=train_config['fp16'],
            bf16=train_config['bf16'],
            optim=train_config['optim'],
            gradient_checkpointing=train_config['gradient_checkpointing'],
            max_grad_norm=train_config['max_grad_norm'],
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to="none"
        )
        
        return training_args
    
    def train(self):
        """Execute the training process."""
        print("=" * 50)
        print("Starting 911 Emergency Dispatcher Model Training")
        print("=" * 50)
        
        # Setup model, tokenizer, and LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Prepare datasets
        self.prepare_datasets()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets.get('validation'),
            tokenizer=self.tokenizer
        )
        
        # Train the model
        print("\nStarting training...")
        trainer.train()
        
        # Save the final model
        output_dir = self.config['training']['output_dir']
        final_model_path = os.path.join(output_dir, "final_model")
        
        print(f"\nSaving final model to {final_model_path}")
        trainer.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print("=" * 50)
        print("Training completed successfully!")
        print("=" * 50)


def main():
    """Main training function."""
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: No GPU detected. Training will be very slow on CPU.")
    
    # Initialize and run trainer
    trainer = EmergencyDispatcherTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
