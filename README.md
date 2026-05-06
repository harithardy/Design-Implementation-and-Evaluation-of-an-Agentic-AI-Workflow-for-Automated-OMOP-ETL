# Agentic AI Pipeline for OMOP ETL

## Overview
This project presents an agentic AI-driven ETL pipeline that automates the transformation of synthetic EHR data into the OMOP Common Data Model (CDM v5.4). The goal is to improve data quality, reproducibility, and auditability compared to traditional manual ETL workflows.

## Objectives
- Automate OMOP ETL using a multi-agent architecture  
- Reduce manual effort and inconsistency  
- Ensure high data quality through validation-driven processing  
- Compare manual vs agentic ETL performance  

## ⚙️ Tech Stack
- Python (pandas, SQLAlchemy)  
- LangGraph (agent orchestration)  
- SQL Server (OMOP database)  
- OHDSI Tools: Data Quality Dashboard, Achilles, ATLAS  
- Standards: OMOP CDM, SNOMED CT, RxNorm, LOINC  

## Key Features
- Agentic workflow using LangGraph  
- Validation-first ETL design  
- Traceable and auditable outputs  
- Standardized vocabulary mapping  
- Modular and reusable architecture  

## Results
- ~99% Data Quality Dashboard (DQD) pass rate 
- Improved consistency over manual ETL  
- Reduced manual effort and debugging  
- Reproducible and scalable pipeline  

## Manual vs Agentic ETL

| Aspect        | Manual ETL   | Agentic ETL     |
|--------------|-------------|----------------|
| Process       | Script-heavy | Automated agents |
| Validation    | Late-stage   | Continuous      |
| Consistency   | Variable     | Standardized    |
| Scalability   | Limited      | High            |


