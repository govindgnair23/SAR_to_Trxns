# SAR to Transactions

Financial Institutions are required to report suspicious activity to law enforcement using SARs (Suspicious Activity Reports).
This is an ongoing Python project to transform Suspicious Activity Reports (SARs) into structured transactions using agentic workflows. The extracted transaction data can be used to:

1)  Backtest Transaction Monitoring Systems using a Simulator
2)  Train ML Models on historical SARs
3)  Build a Knowledge Graph of Historical Suspicious Activities

---

## 🧠 Project Overview

This project uses a sophisticated **multi-agent AI system** to transform unstructured SAR narratives into structured transaction data. The system employs specialized AI agents working in coordinated workflows to:

- Parse and interpret complex SAR narratives
- Extract entities (individuals, organizations, financial institutions, accounts)
- Generate structured transaction records with complete metadata
- Support parallel processing for large-scale SAR analysis

### Key Capabilities
- **Entity Resolution**: Automatically identifies and resolves entity references across complex narratives
- **Transaction Synthesis**: Converts narrative descriptions into structured transaction records
- **Parallel Processing**: Handles multiple SAR documents simultaneously
- **Evaluation Framework**: Built-in metrics and validation for accuracy assessment

---

## 🏗️ Architecture Overview

The system operates through **two coordinated workflows**:

### Workflow 1: Entity Extraction & Resolution
```
SAR Narrative → Entity_Extraction_Agent → Entity_Resolution_Agent → Narrative_Extraction_Agent
```

1. **Entity_Extraction_Agent**: Identifies individuals, organizations, financial institutions, account IDs, and locations
2. **Entity_Resolution_Agent**: Maps accounts to customer IDs and financial institutions  
3. **Narrative_Extraction_Agent**: Creates account-specific sub-narratives for transaction extraction

### Workflow 2: Transaction Generation
```
Sub-Narratives → Router_Agent → Transaction_Generation_Agent → Structured Transactions
```

1. **Router_Agent**: Routes narratives to appropriate transaction generation agents
2. **Transaction_Generation_Agent**: Synthesizes structured transactions with complete metadata
3. **Parallel Processing**: Handles multiple sub-narratives concurrently for performance

### Data Flow
```
Raw SAR Text → Entities & Relationships → Sub-Narratives → Transaction Records → CSV/JSON Output
```

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

### 3. Configure Environment Variables

Create a `.env` file in the root directory with your OpenAI API key:

```bash
# .env file
OPEN_API_KEY=your_openai_api_key_here
```

**Security Note**: Never commit your `.env` file to version control. The `.gitignore` file should already exclude it.

### 4. Configure Agents (Optional)

The agent configurations are stored in `configs/agents_config.yaml`. You can modify:
- Model types (gpt-4o-mini, gpt-4.1, etc.)
- Temperature settings
- System prompts
- Agent behavior parameters

### 5. Run the project

**Command Line Interface:**
```bash
python main.py data/input/sar_test_01.txt
```

**Web Interface:**
```bash
streamlit run ui.py
```

The web interface provides an interactive way to upload SAR files and visualize results.

---

## 🧪 Running Tests

This project uses the built-in `unittest` framework.

To run all tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

**Run Evaluations:**
```bash
# Evaluate Workflow 1 (Entity Extraction)
python evals/eval_workflow1.py

# Evaluate Workflow 2 (Transaction Generation)  
python evals/eval_workflow2.py
```

---

## 📊 Input & Output Formats

### Input Format
The system accepts SAR narrative text files. Example structure:

```text
Investigation case number: B7845120. Michael Smith, the owner of XYZ Consulting LLC, 
is suspected of engaging in suspicious wire transfer activities...

Between February 1, 2023, and May 15, 2023, Smith initiated 15 wire transfers 
totaling $450,000 from the business account (#56789-1234) and 10 wire transfers 
totaling $300,000 from his personal account (#67890-4321)...
```

### Output Format
The system generates structured transaction data in CSV format:

| Transaction_ID | Originator_Name | Originator_Account_ID | Beneficiary_Name | Trxn_Amount | Trxn_Date | Trxn_Channel |
|----------------|-----------------|----------------------|------------------|-------------|-----------|--------------|
| 1 | Michael Smith | 56789-1234 | Unknown | 50000 | 2023-02-01 | Wire |
| 2 | Michael Smith | 67890-4321 | Unknown | 30000 | 2023-02-15 | Wire |

**Complete Fields:**
- `Originator_Name`, `Originator_Account_ID`, `Originator_Customer_ID`
- `Beneficiary_Name`, `Beneficiary_Account_ID`, `Beneficiary_Customer_ID`
- `Trxn_Channel`, `Trxn_Date`, `Trxn_Amount`, `Branch_or_ATM_Location`

---

## 🧰 Tech Stack

- **Language**: Python 3.8+
- **AI Framework**: AutoGen (Multi-agent orchestration)
- **ML Models**: OpenAI GPT-4.1, GPT-4o-mini
- **Data Processing**: Pandas, NumPy
- **Web Interface**: Streamlit
- **Configuration**: PyYAML, python-dotenv
- **Visualization**: NetworkX, Pyvis
- **Testing**: `unittest`
- **Version Control**: Git + GitHub

---

## 🔍 Evaluation Framework

The project includes comprehensive evaluation workflows:

### Workflow 1 Evaluation
- **Entity Metrics**: Precision, recall, F1-score for entity extraction
- **Account Mapping**: Accuracy of account-to-customer relationships
- **Output**: `data/output/evals/workflow1/results_entity_metrics_*.csv`

### Workflow 2 Evaluation  
- **Transaction Metrics**: Count accuracy, amount precision, date validation
- **Completeness**: Field population rates
- **Output**: `data/output/evals/workflow2/results_trxn_metrics_*.csv`

**Run Evaluations:**
```bash
# Interactive evaluation with UI
python evals/eval_workflow1_ui.py
python evals/eval_workflow2_ui.py
```

---

## ✅ Features

- ✅ **Multi-Agent AI System**: Specialized agents for entity extraction, resolution, and transaction generation
- ✅ **Two-Workflow Architecture**: Coordinated pipelines for comprehensive SAR processing
- ✅ **Parallel Processing**: Concurrent handling of multiple sub-narratives for performance
- ✅ **Entity Resolution**: Advanced mapping of accounts, customers, and financial institutions
- ✅ **Structured Output**: Complete transaction records with metadata fields
- ✅ **Web Interface**: Streamlit-based UI for interactive SAR processing
- ✅ **Comprehensive Evaluation**: Built-in metrics and validation frameworks
- ✅ **Configurable Agents**: YAML-based configuration for model selection and behavior
- ✅ **Security Best Practices**: Environment-based API key management

---

## 🔒 Security & Compliance

This tool is designed for **defensive security purposes only**:
- ✅ Financial compliance and anti-money laundering (AML) analysis
- ✅ Transaction monitoring system validation
- ✅ Historical SAR data analysis for regulatory purposes
- ❌ **Not for**: Creating, modifying, or improving malicious code

**Data Security:**
- Store API keys in environment variables, never in code
- SAR data contains sensitive information - follow your organization's data handling policies
- Generated transaction data should be treated as confidential

---

## 📌 Roadmap

**Completed:**
- [x] Multi-agent architecture with specialized roles
- [x] Parallel processing capabilities
- [x] Web interface integration
- [x] Comprehensive evaluation framework

**In Progress:**
- [ ] Support for tabular SAR formats
- [ ] Enhanced entity linking across documents
- [ ] Performance optimization for large-scale processing
- [ ] Additional output formats (JSON, XML)

**Future Enhancements:**
- [ ] Real-time SAR processing API
- [ ] Integration with common AML platforms
- [ ] Advanced visualization dashboards
- [ ] Multi-language SAR support

---


