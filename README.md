# SAR to Trxns

Financial Institutions are required to report suspicious activity to law enforcement using SARs (Suspicious Actiity Reports).
This is an ongoing Python project to transform Suspicious Activity Reports (SARs) into structured transactions using agentic workflows. The extracted transaction can be used to:

1)  Backtest Transaction Monitoring Systems using a Simulator
2)  Train ML Models on historical SARs
3)  Build a Knowledge Graph of Historical SAs

---

## 🧠 Project Overview

This project focuses on:

- Parsing and interpreting SAR narratives.
- Extracting relevant transaction-like data points.
- Converting unstructured reports into a structured format for downstream applications.

---

## 📁 Directory Structure

```
SAR_NARRATIVES_TO_TRXNS/
├── agents/              # Agent logic (LLMs or rules)
├── configs/             # Config files for parameters, paths, etc.
├── data/                # Input or processed datasets
├── evals/               # Evaluation scripts or results
├── experiments/         # Experiments and test runs
├── temp/                # Temporary or intermediate files
├── tests/               # Unit tests (using unittest)
├── venv/                # Python virtual environment
├── .gitignore
├── main.py              # Entry point
├── README.md
├── requirements.txt
└── utils.py             # Helper functions

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SAR-to-Trxns.git
cd SAR-to-Trxns
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py <Name of SAR to be extracted>
```

---

## 🧪 Running Tests

This project uses the built-in `unittest` framework.

To run all tests:

```bash
python -m unittest discover -s experiments/tests -p 'test_*.py'
```

---

## 🧰 Tech Stack

- **Language**: Python
- **Testing**: `unittest`
- **Version Control**: Git + GitHub

---

## ✅ Features

- ✅ Narrative-to-transaction transformation  
- ✅ Modular pipeline for easy testing  
- ✅ Unit tests with `unittest`  
- ✅ Data-driven architecture  

---

## 📌 TODO

- [ ] Add support for additional SAR formats including Tables
- [ ] Add Reflection for Agents 
- [ ] Integrate with a frontend or API
- [ ]  Parallelize processing 

---


