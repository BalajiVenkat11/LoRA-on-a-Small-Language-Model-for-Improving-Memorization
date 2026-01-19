# 🏏 Fine-Tuning Qwen2.5 for Cricket Statistics Recall

This repository explores the limits of **Knowledge Injection** in Small Language Models (SLMs). Specifically, we test whether **Qwen2.5-1.5B-Instruct** can memorize and recall exact numerical cricket statistics using **LoRA (Low-Rank Adaptation)** without the aid of Retrieval-Augmented Generation (RAG).

## 🔍 Problem Statement
Can a 1.5B parameter model be transformed into a reliable statistical database through supervised fine-tuning? 

While LLMs excel at style transfer and reasoning, storing high-entropy numerical data (e.g., "11,953 runs") within neural weights is a significant challenge. This project documents the "Capacity Ceiling" encountered when attempting to force-memorize ~2,700 unique statistical data points.

---

## 📊 Dataset & Preparation
- **Source:** Kaggle – Cricket Statistics for All Formats (Focus: Test Batting).
- **Format:** Tabular `clean_tb.csv` converted into Question-Answer pairs.
- **Prompt Augmentation:** To ensure robustness, each statistic was mapped to 3 distinct prompt variations (e.g., *"How many runs..."* vs *"Total runs scored by..."*).
- **Volume:** ~2,700 total QA data points stored in JSONL.

---

## 🧠 Training Configuration
The model was trained on a **Google Colab / Kaggle T4 GPU** (15GB VRAM) with the following PEFT/LoRA parameters:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | Qwen2.5-1.5B-Instruct | Lightweight SLM with strong reasoning. |
| **Method** | QLoRA (4-bit) | Essential for fitting in 15GB VRAM. |
| **LoRA Rank ($r$)** | 8 | Standard for task/style adaptation. |
| **LoRA Alpha** | 16 | Scaling factor (2x Rank). |
| **Target Modules** | `q_proj`, `v_proj` | Targeted Attention layers. |
| **Epochs** | 3 | Standard training duration. |
| **Learning Rate** | 3e-4 | Standard AdamW rate. |



---

## 📈 Evaluation Results
The model was tested on the **Training Set** to evaluate its maximum memorization capacity (Recall).
Hence, the general idea of training-validation set won't hold true here.

| Metric | Result |
| :--- | :--- |
| **Exact Match (Train)** | **~0.18%** |
| **Exact Match (Validation)** | **~0.0%** |
| **Format Accuracy** | **~85%** |

### Sample Outputs vs. Ground Truth
| Prompt | Model Output | Golden (Target) | Result |
| :--- | :--- | :--- | :--- |
| Runs by BC Lara in Tests? | 10,675 | **11,953** | ❌ Skewed |
| Test average of SPD Smith? | 51.42 | **61.99** | ❌ Hallucinated |
| Test career span of DL Haynes?| 1976–1989 | **1978–1994** | ✅ Pattern Correct |

---

## 💡 Key Technical Inferences

### 1. Pattern vs. Fact (The "Year" Success)
The model successfully recalled "Year Spans" (YYYY-YYYY) with reasonable accuracy. 
* **Observation:** Years follow a low-entropy, predictable structural pattern that the model's pre-existing weights already understand.
* **Inference:** Fine-tuning is significantly better at **template-filling** (learning the shape of data) than **knowledge-storing** (learning the specific values).

### 2. The LoRA Capacity Bottleneck
With $r=8$ and targeting only `q_proj`/`v_proj`, the adapter parameters (~1.1M) were too few.
* **Observation:** Factual knowledge is typically localized in the **Feed-Forward Networks (FFN)** rather than the Attention layers.
* **Inference:** By not targeting `up_proj` or `down_proj`, the model lacked the "memory slots" required for dense data storage.



### 3. The "Augmentation Tax"
Using 3 prompt variations per fact was necessary for usability but hindered memorization.
* **Inference:** In low-parameter regimes, prompt variety dilutes the gradient signal. The model uses its limited capacity to learn the **linguistic variety** of the questions rather than the **numerical precision** of the answers.

### 4. Tokenization of Numbers
* **Observation:** Numerical stats are often split into multiple tokens (e.g., `18,426` → `[18]`, `[42]`, `[6]`).
* **Inference:** For a 1.5B model, missing a single token in the sequence results in a "factually wrong" answer, even if the result is numerically close.



---

## 🚀 Future Work: How to Improve
To improve factual recall in Version 2.0, the following changes are recommended:
1. **Increase Rank ($r$):** Move to $r=128$ or $r=256$ for higher storage capacity.
2. **Target All Layers:** Include `gate_proj`, `up_proj`, and `down_proj` (All-Linear LoRA).
3. **Zero Dropout:** Set `lora_dropout=0` to prevent intentional "forgetting" during training.
4. **Repetition:** Increase to 10–15 epochs to force the weights to settle on specific numerical values.

## 🧾 Conclusion
This experiment confirms that **LLMs are reasoning engines, not databases.** While they can learn the *format* of structured data, they struggle to internalize *exact values* through low-rank fine-tuning. For applications requiring 100% numerical accuracy, **RAG (Retrieval-Augmented Generation)** remains the superior architectural choice.

---
