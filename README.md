# AI Financial Reporting Pipeline

Production-grade AI-assisted financial reporting and close automation system.

This project implements a modular multi-agent financial reporting architecture capable of:

* Trial Balance normalization
* AI-assisted account mapping
* FX translation
* Manual journal adjustments
* Financial statement generation
* Cash Flow Statement generation
* Statement of Changes in Equity
* AI anomaly detection
* CFO-style narrative generation
* Immutable audit trails

The architecture combines deterministic accounting logic with controlled LLM reasoning to ensure auditability, explainability, and financial correctness.

---

# Architecture Overview

Pipeline flow:

```text
Raw TB
  ↓
MapperAgent
  ↓
TranslatorAgent
  ↓
AdjusterAgent
  ↓
AnomalyDetectionAgent
  ↓
StatementBuilder
  ↓
CashFlowAgent
  ↓
ChangesInEquityAgent
  ↓
NarrativeAgent
```

## Agent Responsibilities

| Agent                 | Responsibility                          |
| --------------------- | --------------------------------------- |
| MapperAgent           | Normalize and map accounts to COA       |
| TranslatorAgent       | FX translation into functional currency |
| AdjusterAgent         | Apply validated journal entries         |
| AnomalyDetectionAgent | Detect and explain abnormal balances    |
| StatementBuilder      | Build P&L and Balance Sheet             |
| CashFlowAgent         | Generate indirect-method cash flow      |
| ChangesInEquityAgent  | Generate SOCIE                          |
| NarrativeAgent        | Generate AI financial commentary        |

---

# Key Design Principles

## Deterministic Accounting

The following are handled entirely through deterministic code:

* Statement arithmetic
* Journal balancing
* FX calculations
* Rollups
* Reconciliations
* Validation checks

## Controlled AI Usage

LLMs are only used where semantic interpretation is required:

* Unknown account classification
* Narrative generation
* Anomaly explanation

This minimizes hallucination risk.

## Auditability

Every transformation generates immutable audit events.

Any balance sheet value can be traced back to:

```text
Statement Line
→ Ledger Entry
→ FX Translation
→ Adjustment
→ Source TB Row
```

---

# Features

## Financial Statements

* Profit & Loss
* Balance Sheet
* Cash Flow Statement
* Statement of Changes in Equity

## AI Capabilities

* AI account mapping
* AI anomaly explanations
* AI CFO narrative generation

## Validation Framework

* Balance Sheet reconciliation
* Cash reconciliation
* Equity reconciliation
* Journal validation
* FX completeness checks

## Audit Features

* Immutable event log
* Event lineage
* Source traceability
* Compliance warnings

---

# Project Structure

```text
project/
│
├── pipeline.py
├── tb.csv
├── coa.csv
├── fx.csv
├── prior.csv
├── adjustments.json
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd financial-reporting-pipeline
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
openai
```

---

# OpenAI Configuration

Set your OpenAI API key:

### Linux / macOS

```bash
export OPENAI_API_KEY="your_api_key"
```

### Windows CMD

```cmd
set OPENAI_API_KEY=your_api_key
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

---

# Input Files

## 1. Trial Balance (`tb.csv`)

Example:

```csv
account_code,account_name,currency,debit,credit
1110,Cash,USD,100000,0
2100,Accounts Payable,USD,0,45000
```

---

## 2. Chart of Accounts (`coa.csv`)

Example:

```csv
account_code,account_name,account_type,statement,normal_balance,cf_category
1110,Cash,Asset,BS,Debit,Operating
4100,Revenue,Revenue,PL,Credit,Operating
```

---

## 3. FX Rates (`fx.csv`)

Example:

```csv
currency,rate_type,rate
EUR,period_end,1.08
EUR,period_average,1.05
```

---

## 4. Manual Adjustments (`adjustments.json`)

Example:

```json
{
  "entries": [
    {
      "id": "JE-001",
      "lines": [
        {
          "account": "6100",
          "debit": 1000,
          "credit": 0
        },
        {
          "account": "2100",
          "debit": 0,
          "credit": 1000
        }
      ]
    }
  ]
}
```

---

## 5. Prior Period Trial Balance (`prior.csv`)

Used for:

* Cash Flow Statement
* Equity rollforward
* Reconciliation

---

# Running the Pipeline

Execute:

```bash
python pipeline.py
```

---

# Output

The pipeline generates:

## Console Financial Statements

* Profit & Loss
* Balance Sheet
* Cash Flow Statement
* Statement of Changes in Equity

## AI Narrative

Example:

```text
Revenue growth remained strong during Q4 while operating
expenses increased moderately...
```

## Anomaly Flags

Example:

```text
[1210] Goodwill
Amount: $12,000,000
Explanation: Large intangible balance may require impairment review.
```

## Audit Summary

Example:

```text
Total events: 182
Ledger rows: 93
Accounts: 48
```

---

# Validation Logic

## Journal Validation

Checks:

* Debits = Credits
* Accounts exist in COA

Invalid journals are rejected.

---

## FX Validation

Checks:

* Required rates exist
* Fallback usage logged

Missing rates raise explicit exceptions.

---

## Statement Reconciliation

Checks:

```text
Assets = Liabilities + Equity
```

Cash flow and equity reconciliations are also validated.

---

# Failure Handling

| Failure                      | Handling                              |
| ---------------------------- | ------------------------------------- |
| Hallucinated account mapping | Confidence thresholds + manual review |
| Missing FX rate              | Exception raised                      |
| Invalid journal              | Rejected                              |
| Unknown account              | Quarantined                           |
| Narrative failure            | Graceful fallback                     |
| Reconciliation gap           | Logged warning                        |

---

# Auditability

Each ledger movement creates immutable audit events.

Event example:

```json
{
  "event_type": "FX_TRANSLATION",
  "account_code": "1110",
  "amount": 100000,
  "currency": "USD"
}
```

Auditors can trace every statement value back to source data.

---

# Future Enhancements

Planned enterprise extensions:

* Multi-entity consolidation
* Intercompany eliminations
* Lease accounting
* Deferred tax engine
* ML anomaly detection
* Embedding-based account mapping
* Event-driven processing
* Human review workflows
* Real-time close monitoring

---

# Recommended Production Stack

| Layer           | Technology              |
| --------------- | ----------------------- |
| Compute         | Python                  |
| Data Processing | Pandas / Polars         |
| AI              | OpenAI GPT-4.1          |
| Storage         | PostgreSQL / Snowflake  |
| Event Streaming | Kafka                   |
| Vector Search   | Elasticsearch / LanceDB |
| Orchestration   | LangGraph               |
| APIs            | FastAPI                 |

---

# Security Considerations

For production deployment:

* Store secrets in Vault/KeyVault
* Encrypt financial data
* Enable audit logging
* Implement RBAC
* Add model output validation
* Use private networking

---

# License

MIT License

---

# Author

AI Financial Reporting Pipeline
Production-grade accounting intelligence system
