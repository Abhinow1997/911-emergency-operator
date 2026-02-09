# Example Usage Guide

This guide provides practical examples for using the 911 Emergency Operator project.

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
bash setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create a JSONL file with your 911 call transcripts at `data/911_calls.jsonl`:

```json
{"caller": "Emergency message here", "dispatcher": "Dispatcher response", "emergency_type": "type", "location": "address"}
```

For demonstration, sample data is provided.

### 3. Train the Model

```bash
python train.py
```

Training will:
- Load Gemma-2B-IT with 4-bit quantization
- Apply LoRA adapters
- Train on your dataset
- Save the model to `outputs/final_model`

### 4. Test the Model

Interactive mode:
```bash
python inference.py --interactive
```

Single message:
```bash
python inference.py --message "There's a fire in my building!"
```

## Advanced Usage

### Custom Configuration

Edit `config.yaml` to customize:

```yaml
# Adjust LoRA rank for more/less parameters
lora:
  r: 32  # Higher = more parameters, better quality but slower

# Change training parameters
training:
  num_train_epochs: 5  # More epochs
  per_device_train_batch_size: 2  # Larger batch if you have memory
  learning_rate: 1.0e-4  # Lower learning rate for fine-tuning
```

### Using Your Own Dataset

Format your data as JSONL:

```python
import json

calls = [
    {
        "caller": "Caller's emergency message",
        "dispatcher": "Dispatcher's response",
        "emergency_type": "medical|fire|police|other",
        "location": "Address or location description"
    }
]

with open('data/911_calls.jsonl', 'w') as f:
    for call in calls:
        f.write(json.dumps(call) + '\n')
```

### Memory Optimization

If you run out of memory:

1. Reduce batch size in `config.yaml`:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

2. Enable more aggressive quantization:
```yaml
model:
  max_length: 256  # Reduce from 512
```

### Monitoring Training

The training script outputs:
- Loss values every 10 steps
- Model saves every 100 steps
- Final model at completion

Watch for:
- Loss should decrease over time
- Memory usage should stay stable
- No CUDA out of memory errors

## Example Scenarios

### Medical Emergency
```
Caller: My husband is having chest pains and trouble breathing!
Dispatcher: I'm sending paramedics right away. Is he conscious? Keep him calm and don't let him exert himself.
```

### Fire Emergency
```
Caller: There's smoke coming from my neighbor's apartment!
Dispatcher: I'm dispatching fire services now. Have you evacuated the building? Do not enter the apartment.
```

### Police Emergency
```
Caller: Someone just stole my car from the parking lot!
Dispatcher: I'm sending officers to your location. Can you provide a description of the vehicle and the suspect?
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size to 1
- Reduce max_length to 256
- Enable gradient checkpointing (already enabled by default)

### Model not loading
- Ensure you have completed training
- Check that `outputs/final_model` exists
- Verify you have enough disk space

### Poor quality responses
- Increase training epochs
- Use more diverse training data
- Adjust LoRA rank (higher = better quality)
- Lower learning rate for fine-tuning

## Performance Tips

1. **Use a GPU**: CPU training will take days
2. **Batch Size**: Start with 1, increase if memory allows
3. **Epochs**: 3-5 epochs usually sufficient
4. **Data Quality**: Better training data = better results
5. **LoRA Rank**: 16-32 provides good balance

## Next Steps

After basic training works:
1. Expand your dataset to 518+ transcripts
2. Fine-tune hyperparameters
3. Test on diverse emergency scenarios
4. Evaluate response quality
5. Iterate on data and training configuration
