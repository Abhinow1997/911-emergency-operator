# 911 Emergency Operator - AI Dispatcher Simulation

This project investigates the feasibility of specializing a small, general-purpose instruction-following model—specifically Google's **Gemma-2B-IT**—to simulate the specific communication protocols of a 911 emergency dispatcher. By leveraging **4-bit quantization** and **Low-Rank Adaptation (LoRA)**, the model was successfully fine-tuned on consumer-grade hardware (Tesla T4 GPU) using a dataset of **518 real 911 call transcripts**.

## Project Overview

Emergency dispatch requires specialized communication skills, including:
- Calm and clear communication under pressure
- Protocol-appropriate questioning and responses
- Efficient information gathering
- Appropriate empathy and reassurance
- Quick assessment and prioritization

This project demonstrates that a small 2B parameter model can be effectively specialized for this domain through parameter-efficient fine-tuning techniques.

## Key Features

- **4-bit Quantization**: Uses BitsAndBytes NF4 quantization to reduce memory footprint
- **LoRA Fine-tuning**: Parameter-efficient adaptation with minimal memory overhead
- **Consumer Hardware**: Designed to run on Tesla T4 GPU (16GB VRAM)
- **Real Transcripts**: Trained on 518 real 911 call transcripts
- **Instruction Following**: Formatted for instruction-following task structure

## Technical Architecture

### Model Specifications
- **Base Model**: Google Gemma-2B-IT (Instruction-Tuned)
- **Quantization**: 4-bit NF4 with double quantization
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank (r): 16
  - Alpha: 32
  - Target modules: Query, Key, Value, Output, Gate, Up, Down projections
  - Dropout: 0.05

### Training Configuration
- **Dataset Size**: 518 call transcripts
- **Hardware**: Tesla T4 GPU (or equivalent)
- **Batch Size**: 1 per device with gradient accumulation (4 steps)
- **Learning Rate**: 2e-4
- **Optimizer**: Paged AdamW 32-bit
- **Epochs**: 3
- **Memory Optimization**: Gradient checkpointing enabled

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 20GB+ disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Abhinow1997/911-emergency-operator.git
cd 911-emergency-operator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create sample dataset:
```bash
python data_preparation.py
```

## Usage

### Data Preparation

The model expects data in JSONL format. Each line should contain a JSON object with the following structure:

```json
{
  "caller": "There's been a car accident on Highway 101!",
  "dispatcher": "I'm dispatching emergency services now. Are there any injuries?",
  "emergency_type": "traffic_accident",
  "location": "Highway 101, Exit 25"
}
```

Place your dataset at `data/911_calls.jsonl` or modify the path in `config.yaml`.

### Training

To train the model:

```bash
python train.py
```

The training script will:
1. Load the Gemma-2B-IT base model with 4-bit quantization
2. Apply LoRA adapters
3. Fine-tune on the 911 call transcripts
4. Save the trained model to `outputs/final_model`

### Inference

#### Interactive Mode

Run the model in interactive conversation mode:

```bash
python inference.py --model_path outputs/final_model --interactive
```

This will start an interactive session where you can simulate emergency calls.

#### Single Message Mode

Process a single emergency message:

```bash
python inference.py --model_path outputs/final_model --message "My house is on fire!"
```

## Configuration

All model and training parameters can be customized in `config.yaml`:

- **Model settings**: Model name, max length
- **Quantization**: 4-bit configuration, compute dtype
- **LoRA**: Rank, alpha, target modules, dropout
- **Training**: Batch size, learning rate, epochs, optimization
- **Dataset**: Train/validation split, random seed

## Project Structure

```
911-emergency-operator/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── data_preparation.py      # Data loading and preprocessing
├── train.py                 # Model training script
├── inference.py             # Model inference script
├── data/                    # Dataset directory
│   └── 911_calls.jsonl     # Training data (JSONL format)
└── outputs/                 # Model outputs and checkpoints
    └── final_model/         # Saved fine-tuned model
```

## Hardware Requirements

### Minimum Requirements
- GPU: 16GB VRAM (e.g., Tesla T4, RTX 4060 Ti 16GB)
- RAM: 16GB system memory
- Storage: 20GB free space

### Recommended Requirements
- GPU: Tesla T4, A10, or better
- RAM: 32GB system memory
- Storage: 50GB free space

## Results

The fine-tuned model demonstrates:
- Appropriate emergency dispatcher communication style
- Protocol-aware questioning and responses
- Ability to handle various emergency types (medical, fire, police)
- Calm and clear communication patterns
- Efficient information gathering techniques

## Limitations

- Model is trained on English language transcripts only
- Performance depends on quality and diversity of training data
- Not a replacement for professional 911 dispatchers
- Should be used for research and educational purposes only

## Safety and Ethics

**IMPORTANT**: This model is for research and educational purposes only. It should NOT be used in actual emergency situations. Real 911 services require trained professionals with access to emergency response systems.

## Future Improvements

- [ ] Expand dataset to include more diverse emergency scenarios
- [ ] Add multi-turn conversation capability
- [ ] Implement real-time response evaluation
- [ ] Add support for additional languages
- [ ] Fine-tune on larger models (7B, 13B parameters)
- [ ] Integrate with speech-to-text for voice interaction

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

- Google for the Gemma-2B-IT base model
- HuggingFace for the transformers and PEFT libraries
- BitsAndBytes for efficient quantization implementation

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{911-emergency-operator,
  title={911 Emergency Operator: Fine-tuning Gemma-2B-IT for Emergency Dispatch Simulation},
  author={Abhinow1997},
  year={2024},
  publisher={GitHub},
  howpublished={\\url{https://github.com/Abhinow1997/911-emergency-operator}}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.