# Repository for Journal of Information Security and Applications Paper "Enhancing password security with honeywords and LLMs"

This repository contains all the code and resources necessary for fine-tuning OpenAI's Large Language Models (LLMs) on password datasets, generating secure passwords, evaluating model performance, benchmarking against existing tools, and conducting honeyword analysis.


## 📌 Repository Overview

The main functionalities provided in this repository include:

- ✅ Fine-tuning OpenAI's LLMs on real-world password datasets.
- 🔐 AI-based password generation.
- 📊 Password evaluation using entropy, guess numbers, and statistical analysis.
- 🧪 Benchmarking against traditional tools such as PCFG, Markov, ZXCVBN, and Fuzzy Matching.
- 🐝 Honeyword generation and effectiveness evaluation.
- 📈 Visualizations for model comparisons and results interpretation.

---

## 📁 Directory and Script Summary

| File                                | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `Password_Finetuning.py`           | Fine-tunes OpenAI models on password datasets like RockYou and 4iQ.        |
| `Password_Generation.py`           | Generates passwords from trained LLMs.                                     |
| `Password_Evaluating_and_Plotting.py` | Evaluates and visualizes password strength and entropy distributions.   |
| `Password_Testing.py`              | Benchmarks generated passwords using PCFG, Markov, ZXCVBN, and Fuzzy.     |
| `Honeywords.py`                    | Generates and analyzes honeywords for decoy-based authentication systems.  |

---

## 📂 Datasets Used

### 📌 RockYou Password Dataset: Available on Kaggle:

- Download via: [Kaggle - common-password-list-rockyoutxt](https://www.kaggle.com/datasets/wjburns/common-password-list-rockyoutxt)

### 📌 4iQ Password Dataset: Download instructions and dataset available at:

- Source: [GitHub - tensorflow-1.4-billion-password-analysis](https://github.com/philipperemy/tensorflow-1.4-billion-password-analysis)

### 📌 Honeyword Analysis Dataset: Honeyword simulation code adapted from the paper:  
  *The Impact of Exposed Passwords on Honeyword Efficacy*  
 - Source: [https://github.com/zonghaohuang007/honeywords-analysis](https://github.com/zonghaohuang007/honeywords-analysis)

---

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. Usage
   ```bash
   python <script_name>.py
   ```
   You will need your Open-AI Api key.

## License
This repository is released under the MIT License.
