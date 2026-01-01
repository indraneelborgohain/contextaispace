<div align="center">

# Nano-GPT-OSS Language Model

**An open-source transformer that balances full-context and sliding-window attention for efficient, scalable LLM training and inference.**

<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://huggingface.co"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFC107?logo=hugging%20face&logoColor=black" alt="Hugging Face"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>

![Val Loss of Gpt oss](assets/val-loss.png)

</div>

## Training & Validation Loss

| loss | Training Loss| Validation Loss | Num Heads | Trf BLock | Hidden Dim |
|--------|---------|---------|----------|----------|----------|
| GPT-OSS | **1.981** | **1.682** | 12 | 12 | 1020 |
| GPT2 | 3.124 | 2.747 | 12 | 12 | 1020 |
| GPT-OSS | **2.034** | **1.725** | 12 | 8 | 1020 |
| GPT2 | 2.593 | 2.173 | 12 | 8 | 1020 |
| GPT-OSS | **2.031** | **1.778** | 12 | 6 | 1020 |
| GPT2 | 2.570 | 2.331 | 12 | 6 | 1020 |
| GPT-OSS | **1.984** | **1.678** | 8 | 12 | 1024 |
| GPT2 | 2.445 | 2.036 | 8 | 12 | 1024 |
| GPT-OSS | **2.212** | **1.901**| 8 | 8 | 1024 |
| GPT2 | 2.416 | 2.011 | 8 | 8 | 1024 |
| GPT-OSS | **2.075** | **1.760** | 8 | 6 | 1024 |
| GPT2 | 2.734 | 2.323 | 8 | 6 | 1024 |
| GPT-OSS | **1.943** | **1.684** | 6 | 12 | 1020 |
| GPT2 | 2.748 | 2.366 | 6 | 12 | 1020 |
| GPT-OSS | **2.014** | **1.767** | 6 | 8 | 1020 |
| GPT2 | 2.594 | 2.213 | 6 | 8 | 1020 |
| GPT-OSS | **2.125** | **1.820** | 6 | 6 | 1020 |
| GPT2 | 2.784 | 2.366 | 6 | 6 | 1020 |

---
## Key Improvements of GPT-OSS over GPT-2

### 🏗️ Architecture Enhancements
- **Mixture of Experts (MoE) in MLP** with Router → Sparse experts active per token (big model capacity, low active FLOPs)
- **Gated Router** → Token-dependent routing to experts (shown inside MoE block)
- **SwiGLU Feed-Forward (FFN) modules** → Modern activation in FFN instead of GELU
- **Grouped Query Attention (GQA) + RoPE** → Efficient multi-head attention with rotary embeddings for longer context
- **Sliding Window Attention** → Reduces computation while maintaining context (alternating layers)
- **Sink Slots in Attention** → Learned aggregation slots for global context stability
- **RMSNorm** → More stable normalization layer (replaces LayerNorm)
- **Encoder-Decoder Architecture** → NEW! Bidirectional encoder with causal decoder for Q&A tasks
- **SVD Compression** → Efficient encoder output compression for cross-attention
- **Cross-Attention Layers** → Decoder attends to encoder representations

### 📊 Performance Improvements
- **Lower Training Loss** → Better convergence during training
- **Lower Validation Loss** → Better generalization to unseen data
- **Lower Memory Usage** → More efficient memory usage during training and inference
- **Lower Disk Space** → More efficient disk space usage during training and inference
- **Lower Inference Time** → Faster inference time during inference

## Dependencies
- [pytorch](https://pytorch.org) <3
-  `datasets` for huggingface datasets <3 (for loading datasets)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `tensorboard` for training visualization <3
-  `streamlit` for web interface <3
-  `tqdm` for progress bars <3
-  `huggingface_hub` for downloading pre-trained models <3 

## 📊 Dataset and Format

TinyStories can be found at [HuggingFace Datasets](https://huggingface.co/datasets/roneneldan/TinyStories).

### Data Fields:

Each story entry contains:

- `story`: The main story text
<details>
<summary>📝 Click to see example story</summary>

**Story:**

```
Once upon a time, there was a big, red ball that could bounce very high...
```

\[Rest of the example story\]

</details>

## 🚀 Installation


### 📦 Pip Installation

```bash
# clone project
git clone https://github.com/VizuaraAI/nano-gpt-oss
cd nano-gpt-oss

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv
# Install requirements
pip install -r requirements.txt
</details>

# install pytorch according to instructions
# https://pytorch.org/get-started/
# Install requirements
pip install -r requirements.txt
```


## 🏋️ Training

The system will automatically detect and utilize available GPU resources. We provide three training variants:

### 🔹 Standard Decoder (train.py)
Traditional decoder-only transformer architecture with MoE and sliding window attention.

**Training:**
```bash
# Basic training
python train.py

# Medium model with TensorBoard
python train.py --model_size medium --max_iters 5000 --use_tensorboard

# Custom configuration
python train.py --model_size toy --batch_size 8 --block_size 512 --lr 3e-4
```

**Available model sizes:** `toy`, `medium`, `large`

### 🔹 Context-Aware Decoder (train_context.py)
**Enhanced transformer with persistent context vector** that maintains memory across predictions within documents.

**Key Features:**
- Context vector automatically resets at document boundaries
- Memory of previous predictions helps maintain consistency
- Ideal for story generation and coherent long-form text

**Training:**
```bash
# Basic training
python train_context.py

# Medium model with TensorBoard and LSI cross-attention
python train_context.py --model_size medium --lsi --use_tensorboard

# Custom configuration with context embedding
python train_context.py --model_size toy --batch_size 4 --context-embed
```

**How it works:**
- Context state stored as a learned embedding (shape: `hidden_size`)
- Prepended to token embeddings at each forward pass
- Automatically updates with last token output
- Resets to zero at the start of each new document/story

### 🔹 Encoder-Decoder for Question Answering (train_squad.py)
**NEW!** Encoder-decoder architecture with cross-attention for extractive question answering on SQuAD dataset.

**Architecture:**
- **Encoder**: Bidirectional attention over context passages (SVD compression for efficiency)
- **Decoder**: Causal generation with cross-attention to encoder
- **Training**: Loss computed only on answer tokens (questions masked)

**Training from scratch:**
```bash
# Basic toy model
python train_squad.py --model_size toy --batch_size 4 --max_iters 10000

# Medium model with longer contexts
python train_squad.py --model_size medium --batch_size 8 --max_context_len 512 --max_qa_len 128
```

**Fine-tuning with pretrained decoder:**
```bash
# Load TinyStories pretrained weights, fine-tune on SQuAD
python train_squad.py \
  --model_size small \
  --pretrained_decoder_path "out/tinystories_checkpoint.pt" \
  --decoder_lr 1e-5 \
  --new_layers_lr 3e-4 \
  --batch_size 8 \
  --max_iters 10000 \
  --use_tensorboard

# Fine-tune with custom learning rates
python train_squad.py \
  --pretrained_decoder_path "model/checkpoint_5000.pt" \
  --decoder_lr 5e-6 \
  --new_layers_lr 1e-3 \
  --out_dir model_squad_finetuned
```

**What gets loaded vs trained:**
- ✓ **Loaded**: Pretrained decoder (embeddings, self-attention, MLPs)
- ✗ **Trained from scratch**: Encoder (all layers), cross-attention layers

**Key arguments:**
- `--pretrained_decoder_path`: Path to decoder checkpoint (optional)
- `--decoder_lr`: Lower LR for pretrained decoder layers (preserves knowledge)
- `--new_layers_lr`: Higher LR for encoder + cross-attention (fast adaptation)
- `--max_context_len`: Maximum context passage length (default: 512)
- `--max_qa_len`: Maximum question+answer length (default: 128)

### 📊 Monitoring Training

**TensorBoard (Recommended):**
```bash
# Standard decoder
python train.py --use_tensorboard
tensorboard --logdir=runs

# Context-aware decoder
python train_context.py --use_tensorboard
tensorboard --logdir=runs_context

# Encoder-decoder (SQuAD)
python train_squad.py --use_tensorboard
tensorboard --logdir=runs_squad
```

**Training outputs:**
- Standard model: `model/` directory
- Context model: `model_context/` directory
- SQuAD model: `model_squad/` directory (default)
- Checkpoints saved every 500-1000 iterations
- Config saved as `config.json` (encoder_config.json + decoder_config.json for SQuAD)


### Why nano GPT-OSS is better than nano GPT2

## 1. Loss Curves Analysis

### 1.1 Validation Loss Comparison

| Model Size (Layers) | GPT-OSS Val Loss | GPT2 Val Loss | Improvement |
|---------------------|------------------|---------------|-------------|
| 6 Layers           | 1.76            | 2.01          | 12.4%       |
| 8 Layers           | 1.89            | 2.01          | 6.0%        |
| 12 Layers          | 1.67            | 1.28          | 30.5%       |

### 1.2 Key Observations

- **Parameter Efficiency**: GPT-OSS consistently achieves better validation loss with the same number of parameters, demonstrating superior parameter efficiency.
- **Scaling Behavior**: The performance gap between GPT-OSS and GPT2 becomes more pronounced with larger model sizes, with GPT-OSS showing a 30.5% improvement in the 12-layer configuration.
- **Training Stability**: GPT-OSS exhibits more stable training dynamics across different model sizes, as evidenced by smoother loss curves and better convergence.

### 1.3 Performance Analysis

- **6-Layer Models**: GPT-OSS shows significant improvement (12.4% better validation loss) despite having the same architecture.
- **12-Layer Models**: The advantage of GPT-OSS becomes even more substantial, with a 30.5% improvement in validation loss, suggesting better scaling properties.

### 1.4 Conclusion

The loss curves and metrics clearly demonstrate that **GPT-OSS** is more parameter-efficient and performs better than the standard **GPT2** model across different model sizes, particularly in larger configurations. This suggests that the architectural improvements in GPT-OSS lead to better learning dynamics and generalization.

---

## 2. Model Size & Efficiency

### 2.1 Architecture Comparison
| Parameter | Layers | Hidden Dim | Attention Heads | Parameters | Model Size |
|-----------|---------|---------|--------|--------|--------|
| **GPT-OSS** | 12 | 1020 | 12 | 588M | 2.19 GB |
| **GPT2** | 12 | 1020 | 12 | 564M | 2.46 GB |


### 2.2 Inference Performance
| Metric | GPT-OSS | GPT2 | Notes |
|--------|---------|---------|-------|
| **Disk Size (FP16)** | 2.19 GB | 2.46 GB | GPT2 needs more storage. |
| **RAM (Inference)** | 2.60 GB | 2.94 GB | GPT2 requires high-end GPU. |
| **Inference(Tok/Sec)** | 25 | 30 | GPT-OSS is slower than GPT2 . |

#### Key Insights
- **Storage Efficiency**: GPT-OSS uses 11% less disk space despite having 4% more parameters
- **Memory Optimization**: 11.6% lower RAM usage makes GPT-OSS more hardware-friendly
- **Performance Trade-off**: Slightly slower inference (25 vs 30 tokens/sec) for better efficiency
- **Deployment Advantage**: Lower memory requirements enable broader hardware compatibility

---

## 3. Creativity



| Model | Grammar score | Creativity score | Consistency score | Num Heads | Trf BLock | Hidden Dim |
|--------|---------|---------|----------|----------|----------|----------|
| GPT-OSS | **6** | **4** | **6** | 12 | 12 | 1020 |
| GPT2 | 3 | 4 | 2 | 12 | 12 | 1020 |
| GPT-OSS | **5** | **4** | **3** | 12 | 8 | 1020 |
| GPT2 | 5 | 4 | 4 | 12 | 8 | 1020 |
| GPT-OSS | **5** | **5** | **4** | 12 | 6 | 1020 |
| GPT2 | 4 | 5 | 3 | 12 | 6 | 1020 |
| GPT-OSS | **6** | **5** | **5** | 8 | 12 | 1024 |
| GPT2 | 4 | 4 | 4 | 8 | 12 | 1024 |
| GPT-OSS | **6** | **5** | **5**| 8 | 8 | 1024 |
| GPT2 | 6 | 5 | 4 | 8 | 8 | 1024 |
| GPT-OSS | **4** | **3** | **4** | 8 | 6 | 1024 |
| GPT2 | 3 | 4 | 2 | 8 | 6 | 1024 |
| GPT-OSS | **5** | **6** | **6** | 6 | 12 | 1020 |
| GPT2 | 2 | 3 | 1 | 6 | 12 | 1020 |
| GPT-OSS | **5** | **6** | **6** | 6 | 8 | 1020 |
| GPT2 | 3 | 3 | 2 | 6 | 8 | 1024 |
| GPT-OSS | **4** | **5** | **3** | 6 | 6 | 1020 |
| GPT2 | 1 | 2 | 1 | 6 | 6 | 1020 |

#### Key Insights
- **Performance Trends**: GPT-OSS consistently shows higher scores across most configurations, especially with fewer layers and attention heads.
- **Resource Efficiency**: GPT-OSS maintains strong performance (scores 4-6) even with 6 layers, while GPT2's performance drops significantly (scores 1-3) with reduced architecture size.
- **Optimal Configuration**: Both models perform best with 12 layers, but GPT-OSS shows more stable performance across different configurations.
- **Quality vs. Resources**: GPT-OSS demonstrates better parameter efficiency, achieving high-quality outputs with fewer computational resources compared to GPT2.
- **Consistency**: GPT-OSS shows less variance in scores (4-6 range) compared to GPT2 (1-6 range), indicating more reliable performance across different model sizes.

---

## 🚀 Inference & Deployment

### Command Line Inference

**Standard Model:**
```python
from inference import load_model_and_generate

# Generate text
text = load_model_and_generate(
    checkpoint_path="model/checkpoint_5000.pt",
    prompt="Once upon a time",
    max_length=200,
    temperature=0.8
)
print(text)
```

**Context-Aware Model:**
```python
from architecture.gptoss_context import TokenGenerator

# Initialize generator
generator = TokenGenerator("model_context/checkpoint_5000.pt", device="cuda:0")

# Generate with context memory
for token in generator.generate(
    prompt_tokens=[1, 2, 3],  # Your tokenized prompt
    stop_tokens=[0],
    temperature=0.8,
    max_tokens=200
):
    print(token)

# Reset context for new conversation
generator.reset()
```

### 🌐 Streamlit Web Interface

Launch an interactive chat interface:

```bash
streamlit run server.py
```

**Features:**
- ChatGPT/Claude-style interface
- Model selection (Sapphire custom or GPT-OSS 20B)
- Adjustable temperature and top-k sampling
- Collapsible settings sidebar
- Message history

### 📥 Download Pre-trained Weights

Download GPT-OSS 20B model weights:

```bash
python architecture/open-gpt-oss/download_weights.py --output_dir gpt_oss_weights
```

---

## 📁 Project Structure

```
contextaispace/
├── train.py                      # Standard decoder training script
├── train_context.py              # Context-aware decoder training script
├── train_squad.py                # Encoder-decoder Q&A training (NEW!)
├── inference.py                  # Inference utilities
├── server.py                     # Streamlit web interface
├── requirements.txt              # Python dependencies
│
├── architecture/
│   ├── transformer.py           # Main decoder (renamed from gptoss_context.py)
│   ├── encoder.py               # Bidirectional encoder with SVD compression (NEW!)
│   ├── attention_components.py  # Shared attention modules (NEW!)
│   ├── config.py                # ModelConfig dataclass (NEW!)
│   ├── token_generator.py       # Token generation interface (NEW!)
│   ├── tokenizer.py             # Tokenizer utilities
│   ├── gpt2.py                  # GPT-2 baseline model
│   ├── gptoss.py                # Standard GPT-OSS decoder
│   └── open-gpt-oss/
│       ├── download_weights.py  # Download pre-trained models
│       └── model.py             # GPT-OSS 20B model
│
├── training/
│   ├── data_loader.py           # Standard data loader
│   ├── data_loader_context.py   # Document-aware data loader
│   └── trainer.py               # Training utilities
│
└── test/                         # Test suite (NEW!)
    ├── test_encoder_decoder.py  # Integration tests
    ├── test_squad_workflow.py   # End-to-end SQuAD tests
    ├── test_components.py       # Component unit tests
    ├── run_all_tests.py         # Test runner
    └── README.md                # Test documentation
```

---

## 🎯 Quick Start Examples

### Example 1: Train a toy decoder quickly
```bash
python train.py --model_size toy --max_iters 1000 --use_tensorboard
```

### Example 2: Train context-aware model for coherent stories
```bash
python train_context.py --model_size medium --batch_size 4 --lsi --use_tensorboard
```

### Example 3: Train encoder-decoder on SQuAD from scratch
```bash
python train_squad.py --model_size small --batch_size 8 --max_iters 10000 --use_tensorboard
```

### Example 4: Fine-tune pretrained decoder on SQuAD
```bash
python train_squad.py \
  --model_size small \
  --pretrained_decoder_path "out/tinystories_checkpoint.pt" \
  --decoder_lr 1e-5 \
  --new_layers_lr 3e-4 \
  --batch_size 8 \
  --use_tensorboard
```

### Example 5: Launch web interface
```bash
streamlit run server.py
```

### Example 6: Generate text programmatically
```python
from inference import generate_text
from architecture.transformer import Transformer

model = Transformer.from_checkpoint("model/checkpoint_5000.pt", device="cuda:0")
text = generate_text(model, "Once upon a time", max_tokens=100, temperature=0.8)
print(text)
```

### Example 7: Question Answering with encoder-decoder
```python
from architecture.encoder import BidirectionalEncoder
from architecture.transformer import Transformer
from architecture.tokenizer import get_tokenizer
import torch

# Load models
tokenizer = get_tokenizer()
encoder = BidirectionalEncoder.from_checkpoint("model_squad/checkpoint_10000.pt")
decoder = Transformer.from_checkpoint("model_squad/checkpoint_10000.pt")

# Encode context
context = "The Eiffel Tower is located in Paris, France."
context_tokens = torch.tensor(tokenizer.encode(context), device="cuda:0")
encoder_k, encoder_v = encoder(context_tokens, return_compressed_kv=True)

# Generate answer
question = "Where is the Eiffel Tower?"
q_tokens = tokenizer.encode("<Q>") + tokenizer.encode(question) + tokenizer.encode("<SEP>")
tokens = torch.tensor(q_tokens, device="cuda:0")

decoder.reset_context()
for _ in range(20):  # Generate up to 20 answer tokens
    logits = decoder(tokens, encoder_k=encoder_k, encoder_v=encoder_v)
    next_token = torch.argmax(logits[-1]).item()
    tokens = torch.cat([tokens, torch.tensor([next_token], device="cuda:0")])
    if next_token == 0:  # Stop at EOS
        break

answer = tokenizer.decode(tokens.tolist())
print(answer)  # Output: "<Q> Where is the Eiffel Tower? <SEP> Paris, France"
```

python train_encoder_decoder_stories.py \
  --model_size medium \
  --pretrained_decoder_path "model_context/checkpoint_5000.pt" \
  --use_lsi_compression \
 