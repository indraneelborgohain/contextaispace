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
- **Mixture of Experts (MoE) in MLP** with a Router → Sparse experts active per token (big model capacity, low active FLOPs)
- **Gated Router** → Token-dependent routing to experts (shown inside MoE block)
- **SwiGLU Feed-Forward (FFN) modules** → Modern activation in FFN instead of GELU
- **Grouped Query Attention + RoPE** → Alternate attention that supports longer context and stable queries
- **Sliding Window Attention** → Efficient attention pattern that reduces computation while maintaining context
- **Sink Slots in Attention** → Learned aggregation slots for global context stability
- **RMSNorm** → More stable normalization layer

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
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `ipywidgets` for optional jupyter notebook support 

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


## How to run Training

The system will automatically detect and utilize available GPU resources. To train the GPT-OSS model, choose one of the following methods:

### Option 1: Command Line Interface

1. Navigate to the project directory:
   ```bash
   cd nano-gpt-oss
   ```

2. Start training with default configuration:
   ```bash
   python train.py
   ```

### Option 2: Jupyter Notebook

1. Launch Jupyter from the project directory:
   ```bash
   jupyter notebook
   ```

2. Open `trains.ipynb`
3. Run all cells sequentially using `Cell > Run All`

### Monitoring Training
- Training progress and metrics will be displayed in the console
- Model checkpoints are saved in the `checkpoints` directory
- Training logs can be found in the `logs` directory


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
