# 📘 Sentiment Analysis on Amazon Customer Reviews Using Parallel Deep Learning

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Project Scope](#project-scope)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Experiments & Methodology](#experiments--methodology)
- [Parallel Processing Techniques](#parallel-processing-techniques)
- [Results & Analysis](#results--analysis)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Key Findings](#key-findings)
- [References](#references)

---

## 🎯 Project Overview
This project implements scalable sentiment analysis on **3.5M+ Amazon customer reviews** using parallel data processing and distributed deep learning.  
Techniques such as **Distributed Data Parallelism (DDP)**, **model parallelism**, and **Dask-based preprocessing** were evaluated to measure their impact on **training time, speedup, and efficiency**.

The goal is to understand how parallel computing techniques accelerate large-scale NLP workloads and optimize resource utilization.
(Details sourced from project PDF) 

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
| CPUs | Training Time (min) |
|------|----------------------|
| 8    | 263                  |
| 16   | ~180                 |
| 24   | ~135                 |
| 28   | Higher due to overhead |

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
| Tool   | Time (sec) |
|--------|------------|
| Dask   | ~30        |
| Pandas | ~80        |

Dask demonstrates **~3× faster** preprocessing on large files.

---

## 📁 Repository Structure


