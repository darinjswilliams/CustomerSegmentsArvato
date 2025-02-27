# Identify Customer Segments

## Project Overview
On this project, I applied **unsupervised learning techniques** to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct **marketing campaigns** toward audiences with the highest expected rate of returns. The dataset used in this project is provided by **Bertelsmann Arvato Analytics**, representing a real-life data science task.

## Project Framework
The notebook provides a structured framework for completing the analysis. Each step contains:
- **Task Description**: Explanation of the subtask.
- **Code Cells**: Predefined cells for implementing solutions.
- **Discussion Sections**: Markdown cells to document findings and methodology.

### Key Considerations
- While some tasks have precise guidelines based on Customer request, others task require **independent decision-making**.
- **Clear documentation** of the approach is essential to ensure reproducibility and collaboration with other data scientists.
- The provided code cells outline major tasks, but additional **exploratory analysis** was required to uncover deeper insights.

## Evaluation Criteria
The project was assessed based on:
- **Code Implementation**: Accuracy and efficiency of solutions.
- **Observations & Conclusions**: Clarity in reporting insights and decisions.
- **Data Handling**: Justification of preprocessing steps and segmentation approach.

This project simulates a **real-world data science scenario**, emphasizing both **technical execution** and **effective communication** of findings.

## Data Dictionary for Identify Customer Segments Data

### Introduction
The data for this project consist of two files:
- **Udacity_AZDIAS_Subset.csv**: Demographics data for the general population of Germany; 891,211 persons (rows) x 85 features (columns).
- **Udacity_CUSTOMERS_Subset.csv**: Demographics data for customers of a mail-order company; 191,652 persons (rows) x 85 features (columns).

The columns in the general demographics file and customers data file are the same. This file documents the features that appear in the data files, sorted in order of appearance. Sections of this file are based on the level of measurement of each feature. The file **"AZDIAS_Feature_Summary.csv"** contains a summary of feature attributes, including information level, data type, and codes for missing or unknown values.

### Table of Contents
1. Person-level features
2. Household-level features
3. Building-level features
4. RR4 micro-cell features
5. RR3 micro-cell features
6. Postcode-level features
7. RR1 neighborhood features
8. PLZ8 macro-cell features
9. Community-level features
