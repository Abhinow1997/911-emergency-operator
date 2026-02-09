"""
Utility functions for the 911 Emergency Operator project.
"""

import torch
import os


def check_gpu_availability():
    """
    Check if GPU is available and print information.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU Information")
        print("=" * 60)
        print(f"GPU Available: Yes")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        # Memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        
        if total_memory < 16:
            print("\nWarning: GPU has less than 16GB memory.")
            print("Training may fail due to insufficient memory.")
            print("Consider reducing batch size or using CPU (very slow).")
        
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("GPU Information")
        print("=" * 60)
        print("GPU Available: No")
        print("\nWarning: No GPU detected!")
        print("Training will be extremely slow on CPU.")
        print("Please ensure CUDA is properly installed and a GPU is available.")
        print("=" * 60)
        return False


def estimate_training_time(num_samples: int, batch_size: int, epochs: int,
                          gradient_accumulation_steps: int = 4):
    """
    Estimate training time based on dataset size and configuration.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size per device
        epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        str: Estimated training time message
    """
    # Rough estimates based on typical training speeds
    # These are approximate and vary based on hardware
    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs
    
    # Assume ~2-3 seconds per step on T4 GPU
    estimated_seconds = total_steps * 2.5
    
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    
    message = f"\nEstimated Training Time: "
    if hours > 0:
        message += f"{hours}h {minutes}m"
    else:
        message += f"{minutes}m"
    
    message += f" ({total_steps} total steps)\n"
    
    return message


def print_model_info(model):
    """
    Print information about the model.
    
    Args:
        model: The model to print information about
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("Model Information")
        print("=" * 60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage Trainable: {100 * trainable_params / total_params:.2f}%")
        print("=" * 60)
    except Exception as e:
        print(f"Could not print model info: {e}")


def validate_config(config: dict) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    required_keys = ['model', 'quantization', 'lora', 'training', 'dataset']
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required config section: {key}")
            return False
    
    # Check training dataset path
    dataset_path = config['training'].get('dataset_path')
    if dataset_path and not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found at {dataset_path}")
        print("Sample data will be used for demonstration.")
    
    return True


def format_time(seconds: float) -> str:
    """
    Format seconds into a readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    # Run diagnostics
    check_gpu_availability()
