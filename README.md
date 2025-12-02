# 📘 Sentiment Analysis on Amazon Customer Reviews Using Parallel Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/devmithun7/Sentiment-Analysis-using-LSTM)](https://github.com/devmithun7/Sentiment-Analysis-using-LSTM/stargazers)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Project Scope](#-project-scope)
- [Dataset](#-dataset)
- [Technical Architecture](#️-technical-architecture)
- [Experiments & Methodology](#-experiments--methodology)
- [Parallel Processing Techniques](#-parallel-processing-techniques)
- [Results & Analysis](#-results--analysis)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)
- [References](#-references)

---

## 🎯 Project Overview

This project implements scalable sentiment analysis on **3.5M+ Amazon customer reviews** using parallel data processing and distributed deep learning. Techniques such as **Distributed Data Parallelism (DDP)**, **model parallelism**, and **Dask-based preprocessing** were evaluated to measure their impact on **training time, speedup, and efficiency**.

The goal is to understand how parallel computing techniques accelerate large-scale NLP workloads and optimize resource utilization.

Sentiment-Analysis-using-LSTM/
│
├── 📂 POC notebooks/
│ ├── initial_model_exploration.ipynb # Initial LSTM model experiments
│ ├── data_preprocessing_tests.ipynb # Data cleaning and preprocessing trials
│ ├── model_architecture_comparison.ipynb # LSTM vs GRU performance comparison
│ └── baseline_performance.ipynb # Single-threaded baseline metrics
│
├── 📂 data_and_model_parallel/
│ ├── train_hybrid_parallel.py # Combined data + model parallelism training
│ ├── model_parallel_lstm.py # Model parallelism LSTM implementation
│ ├── distributed_data_loader.py # Custom distributed data loading
│ ├── hybrid_performance_analysis.ipynb # Performance metrics and analysis
│ ├── model.py # Model architecture definitions
│ └── logs/models/metrics/plots/ # Training artifacts and results
│
├── 📂 data_parallel/
│ ├── ddp_training.py # DistributedDataParallel implementation
│ ├── main.py # Main execution script
│ ├── model.py # LSTM model architecture
│ ├── distributed_sampler.py # Custom distributed sampling logic
│ ├── scaling_analysis.ipynb # CPU scaling performance analysis
│ └── logs/models/metrics/plots/ # Training outputs and visualizations
│
├── 📂 data_parallel_and_AMP/
│ ├── amp_ddp_training.py # AMP-enabled distributed training
│ ├── main.py # Main execution script
│ ├── model.py # Memory-optimized LSTM model
│ ├── gradient_scaler.py # Custom gradient scaling utilities
│ ├── memory_usage_analysis.ipynb # Memory consumption analysis
│ └── logs/models/metrics/plots/ # AMP training results
│
├── 📂 dataset/
│ ├── main_data.csv # Primary dataset (1.5GB)
│ ├── subset_data.csv # Subset for testing (163MB)
│ ├── data_loader.py # Custom PyTorch data loader
│ ├── preprocessing_pipeline.py # Dask-based preprocessing pipeline
│ ├── text_cleaning.py # Text cleaning and normalization
│ ├── label_encoding.py # Sentiment label processing
│ └── dataset_info.json # Dataset metadata and statistics
│
├── 📂 SerialProcessing/
│ ├── 📂 cpu/
│ │ ├── SerialExecutionCPU.ipynb # Serial CPU training
│ │ ├── single_thread_lstm.py # Single-threaded LSTM implementation
│ │ └── logs/models/metrics/plots/ # Serial training artifacts
│ └── 📂 gpu/
│ ├── SerialExecutionGPU.ipynb # Serial GPU training
│ ├── SerialExecutionGPU-BatchSize.ipynb # Batch size optimization
│ ├── single_gpu_lstm.py # Single GPU implementation
│ └── logs/models/metrics/plots/ # GPU training results
│
├── 📂 ParallelProcessing/
│ ├── 📂 cpus_with_DDP/
│ │ ├── ddp_train.py # DDP CPU training script
│ │ ├── main.py # Main execution script
│ │ ├── model.py # Model architecture
│ │ ├── ParallelExecutionCPU.ipynb # CPU parallel analysis
│ │ └── logs/models/metrics/plots/ # DDP CPU results
│ │
│ ├── 📂 gpus_with_DDP/
│ │ ├── ddp_train.py # DDP GPU training script
│ │ ├── main.py # Main execution script
│ │ ├── model.py # GPU model implementation
│ │ ├── ParallelExecutionGPU.ipynb # GPU parallel analysis
│ │ └── logs/models/metrics/plots/ # DDP GPU results
│ │
│ ├── 📂 cpus_with_DDP_AMP/
│ │ ├── ddp_train.py # DDP + AMP CPU training script
│ │ ├── main.py # Main execution script
│ │ ├── model.py # AMP-optimized model
│ │ ├── ParallelExecutionCPU_AMP.ipynb # CPU AMP analysis
│ │ └── logs/models/metrics/plots/ # CPU AMP results
│ │
│ └── 📂 gpus_with_DDP_AMP_ModelParallel/
│ ├── ddp_train.py # Full parallelism pipeline
│ ├── main.py # Main execution script
│ ├── model.py # Model parallel architecture
│ ├── FullParallelExecution.ipynb # Complete parallel analysis
│ └── logs/models/metrics/plots/ # Full parallelism results
│
├── 📂 Analysis/
│ ├── CPU-Comparison.ipynb # CPU performance analysis
│ ├── GPU-Comparison.ipynb # GPU performance analysis
│ └── Scalability-Analysis.ipynb # Overall scalability metrics
│
├── 📂 preprocessed_sentiment_data/ # Processed Amazon reviews (not in repo)
│
├── Code Structure.txt # Detailed code organization
├── EDA and Data Analysis.ipynb # Exploratory Data Analysis
├── SpeedUp and Efficiency.ipynb # Performance benchmarking
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## 📁 Repository Structure

---

## 🔬 Project Scope

### **1. Enhance Data Processing Efficiency**

- Parallel ingestion using **Dask**
- Clean and standardize review text (lowercasing, removing non-alphabetic chars)
- Convert multi-class review ratings into binary sentiment labels
- Repartition data for balanced worker workloads
- Generate performance reports (task stream, worker profiles, bandwidth)

### **2. Improve Scalability & Resource Utilization**

- Implement **DistributedDataParallel (DDP)** across multiple CPUs
- Compare scaling behavior for 8, 16, 20, 24, and 28 CPU configurations
- Evaluate communication overhead and bottlenecks

### **3. Optimize Deep Learning Model Training**

- Compare **LSTM** and **GRU**, selecting LSTM for best performance
- Build distributed data loaders with `DistributedSampler`
- Implement model saving and epoch-level evaluation

### **4. Benchmark Performance & Scalability**

- Measure training time reduction with increasing CPUs
- Compute speedup and efficiency metrics
- Evaluate how model parallelism affects performance

---

## 📊 Dataset

### **Dataset Source**

- Amazon Customer Reviews dataset (Kaggle)
- Two files:
  - `main_data` (1.5 GB)
  - `subset_data` (163 MB)

### **Dataset Characteristics**

- Total Records: **3.5M+**
- Labels formatted for fastText:
  - `__label__1` → Negative (1–2 stars)
  - `__label__2` → Positive (4–5 stars)
- Neutral (3-star) reviews excluded
- No missing values
- Final Split:
  - **3,600,000** training samples
  - **400,000** test samples

### **Preprocessing Pipeline**

- Lowercasing text
- Removing non-alphabetic characters
- Stripping unnecessary whitespace
- Partitioning via Dask for parallel efficiency
- Converting labels into binary sentiment

---

## 🏗️ Technical Architecture

### **Model: LSTM-Based Sentiment Classifier**

- Word embeddings
- LSTM encoder
- Fully connected layer with sigmoid output
- Loss: Binary Cross-Entropy
- Optimizer: Adam

### **Distributed Training Setup**

- Backend: **gloo** (CPU)
- Multi-process execution using `torch.multiprocessing.spawn`
- Distributed DataLoader using `DistributedSampler`
- Parallel saving of checkpoints

### **Model Parallel Configuration**

- Embedding + FC on `device0`
- LSTM on `device1`
- Forward pass flow:
  1. Embedding → device0
  2. LSTM → device1
  3. FC layer → device0

---

## 🧪 Experiments & Methodology

### **Training Configurations**

- Epochs: 20
- Batch sizes: distributed across CPU cores
- Hardware: multi-core CPU nodes
- DDP runs: 8, 16, 20, 24, 28 CPUs
- Model Parallel + DDP runs: multiple CPU splits

### **Experimental Goals**

- Identify optimal CPU count for training
- Measure diminishing returns at high CPU counts
- Compare pure DDP vs DDP + Model Parallelism

---

## ⚡ Parallel Processing Techniques

### **1️⃣ Distributed Data Parallel (DDP)**

**Concept:**
Replicate model across processes, each handling unique data shards.

**Implementation Steps:**

- Initialize process group
- Wrap model with `DistributedDataParallel`
- Use `DistributedSampler` for balanced dataset splits
- Synchronize gradients via all-reduce

**Benefits:**

- Near-linear scaling at smaller CPU counts
- No model architecture changes needed

---

### **2️⃣ Model Parallelism**

**Concept:**
Split model layers across multiple CPU devices.

**Device Allocation Example:**

- Device 0: Embedding, FC
- Device 1: LSTM

**Benefits:**

- Better utilization of multiple devices
- Reduces load on single CPU
- Useful for large models

**Trade-offs:**

- Higher communication overhead
- Less efficient if model fits comfortably on one CPU

---

## 📊 Results & Analysis

### **DDP Performance**

| CPUs | Training Time (min) | Speedup | Efficiency |
|------|-------------------|---------|------------|
| 8    | 263               | 1.00x   | 100%       |
| 16   | ~180              | 1.46x   | 91%        |
| 24   | ~135              | 1.95x   | 81%        |
| 28   | Higher due to overhead | <1.95x | <70%    |

**Insights:**

- Training time nearly **cuts in half** from 8 → 24 CPUs
- Best speedup: **1.95×**
- Efficiency decreases with CPU count due to synchronization costs

---

### **DDP + Model Parallel Performance**

- Best performance at **16 CPUs**
- Unstable results at **20 CPUs** due to communication overhead
- Peak speedup: **1.82×**
- Slightly better efficiency at high CPU counts compared to DDP-alone

---

### **Data Preprocessing Performance**

| Tool   | Time (sec) | Speedup |
|--------|------------|---------|
| Dask   | ~30        | 2.67x   |
| Pandas | ~80        | 1.00x   |

Dask demonstrates **~3× faster** preprocessing on large files.

---

## 🚀 Getting Started

### **Prerequisites**

Python 3.8+
PyTorch 1.9+
CUDA 11.0+ (for GPU training)### **Installation**

1. **Clone the repository:**
git clone https://github.com/devmithun7/Sentiment-Analysis-using-LSTM.git
cd Sentiment-Analysis-using-LSTM2. **Install dependencies:**
pip install -r requirements.txt3. **Download dataset:**
# Place Amazon Customer Reviews dataset in dataset/ folder
# Files: main_data.csv, subset_data.csv### **Usage**

#### **Serial Training (Baseline)**
# CPU Serial Training
jupyter notebook SerialProcessing/cpu/SerialExecutionCPU.ipynb

# GPU Serial Training
jupyter notebook SerialProcessing/gpu/SerialExecutionGPU.ipynb#### **Parallel Training**
# DDP CPU Training
cd ParallelProcessing/cpus_with_DDP/
python main.py --epochs 20 --batch-size 64

# DDP GPU Training
cd ParallelProcessing/gpus_with_DDP/
python main.py --epochs 20 --batch-size 128

# Full Parallelism (DDP + AMP + Model Parallel)
cd ParallelProcessing/gpus_with_DDP_AMP_ModelParallel/
python main.py --epochs 20 --amp --model-parallel#### **Analysis**
# Performance Analysis
jupyter notebook Analysis/CPU-Comparison.ipynb
jupyter notebook Analysis/GPU-Comparison.ipynb
jupyter notebook SpeedUp\ and\ Efficiency.ipynb---

## 🔍 Key Findings

### **Performance Insights**

1. **Optimal CPU Count**: 24 CPUs provide the best balance of speedup and efficiency
2. **Diminishing Returns**: Beyond 24 CPUs, communication overhead outweighs benefits
3. **Model Parallelism**: Most effective when combined with moderate DDP scaling
4. **AMP Benefits**: Significant memory savings with minimal accuracy loss

### **Scalability Lessons**

- **Linear scaling** achievable up to 16-20 CPUs
- **Communication bottlenecks** become dominant at high CPU counts
- **Hybrid approaches** (DDP + Model Parallel + AMP) offer best resource utilization

### **Production Recommendations**

- Use **16-24 CPUs** for optimal training efficiency
- Implement **AMP** for memory-constrained environments
- Consider **model parallelism** for very large models
- Monitor **communication overhead** in distributed setups

---

## 📚 References

- [Amazon Customer Reviews Dataset](https://www.kaggle.com/datasets/amazon-customer-reviews)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Dask Parallel Computing](https://dask.org/)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Mithun** - [@devmithun7](https://github.com/devmithun7)

Project Link: [https://github.com/devmithun7/Sentiment-Analysis-using-LSTM](https://github.com/devmithun7/Sentiment-Analysis-using-LSTM)

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!
