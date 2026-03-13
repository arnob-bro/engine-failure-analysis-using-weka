# Cursor Build Instructions
## Project: Engine Failure Detection using Data Mining Techniques with WEKA

This document instructs Cursor (or any coding AI assistant) to build a **Java project using the WEKA Java API** that performs engine failure detection and analysis using multiple data mining techniques.

The project must include:
- Data preprocessing
- Supervised learning
- Unsupervised learning
- Association rule mining
- Model evaluation
- Model persistence

The system will use the **Engine Failure Detection dataset** (synthetic telemetry dataset containing engine sensor readings and fault severity labels).

---

# 1. Technology Stack

Language:
- Java

Libraries:
- WEKA (weka-stable 3.8.x)

Development Environment:
- IntelliJ IDEA / Eclipse

Dataset Format:
- CSV input
- ARFF conversion for WEKA

---

# 2. Project Structure

Create the following project structure:

```
engine-failure-weka/

├── data/
│   ├── engine_failure_dataset.csv
│   └── engine_failure_dataset.arff
│
├── models/
│   └── best_model.model
│
├── src/
│   └── enginefailure/
│       ├── Main.java
│       ├── DataLoader.java
│       ├── Preprocessor.java
│       ├── SupervisedTrainer.java
│       ├── ClusterAnalyzer.java
│       ├── AssociationMiner.java
│       └── ModelPersistence.java
│
└── README.md
```

---

# 3. Dataset Description

The dataset contains engine telemetry features including:

- temperature
- rpm
- fuel_efficiency
- vibration_x
- vibration_y
- vibration_z
- torque
- power_output
- operational_mode (idle / cruising / heavy_load)
- timestamp
- fault_severity (normal / minor_fault / moderate_fault / severe_fault)

The attribute **fault_severity** will be the **class label** for supervised learning.

---

# 4. Required Functional Modules

Cursor must implement the following modules.

---

# 4.1 DataLoader

Responsibilities:

- Load CSV dataset
- Convert CSV → ARFF
- Save ARFF file

Use WEKA classes:

- CSVLoader
- ArffSaver

Output:

```
Instances dataset
```

---

# 4.2 Preprocessor

Apply preprocessing using WEKA Filters.

Required preprocessing steps:

1. Replace missing values
2. Convert string attributes to nominal
3. Normalize numeric attributes

Use filters:

- ReplaceMissingValues
- StringToNominal
- Normalize

Return:

```
Instances processedDataset
```

---

# 4.3 SupervisedTrainer

Train multiple classifiers for engine failure detection.

Algorithms:

- RandomForest
- J48 Decision Tree
- SMO (Support Vector Machine)

Training method:

- Stratified 10‑fold cross validation

Evaluation metrics required:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

Cursor must implement an **evaluation method** using:

```
weka.classifiers.Evaluation
```

The best performing classifier should be selected.

---

# 4.4 ClusterAnalyzer

Apply unsupervised learning to detect patterns in engine telemetry.

Algorithm:

```
SimpleKMeans
```

Configuration:

- Number of clusters: 3 or 4

Output:

- Cluster centroids
- Cluster assignments

Interpretation goal:

Determine if some clusters correspond to **higher fault severity conditions**.

---

# 4.5 AssociationMiner

Discover relationships between engine parameters.

Steps:

1. Discretize numeric attributes
2. Run association rule mining

Filters:

```
Discretize
```

Algorithms:

```
Apriori
```

Parameters:

- Minimum support: 0.1
- Minimum confidence: 0.7

Output:

- Top association rules
- Lift values

Example rules expected:

```
high_temperature ∧ high_vibration ⇒ severe_fault
heavy_load ∧ high_rpm ⇒ moderate_fault
```

---

# 4.6 ModelPersistence

Implement model saving and loading.

Use WEKA utility:

```
weka.core.SerializationHelper
```

Functions required:

Save model

```
SerializationHelper.write()
```

Load model

```
SerializationHelper.read()
```

Saved model location:

```
/models/best_model.model
```

---

# 5. Main Pipeline

Main.java must orchestrate the full workflow.

Pipeline:

```
Load CSV
      ↓
Convert to ARFF
      ↓
Preprocess Dataset
      ↓
Exploratory Summary
      ↓
Supervised Training
      ↓
Model Evaluation
      ↓
Clustering Analysis
      ↓
Association Rule Mining
      ↓
Save Best Model
      ↓
Load Model
      ↓
Run Sample Prediction
```

---

# 6. Output Requirements

The program should print:

1. Dataset summary statistics
2. Classification evaluation results
3. Confusion matrix
4. Cluster results
5. Association rules
6. Prediction result for a sample instance

---

# 7. Additional Requirements

Cursor should ensure:

- Code is modular
- Clear separation of responsibilities
- Use of WEKA Java API instead of GUI
- Proper comments in code
- Error handling for file loading

---

# 8. Example Prediction Flow

```
Load trained model
Input engine telemetry record
Predict fault severity
Output predicted class
```

Example output:

```
Predicted Fault Severity: moderate_fault
```

---

# 9. Project Goal

The final system should demonstrate how **data mining techniques can be applied to engine telemetry data to detect and analyze engine failure conditions**.

The project must showcase three major paradigms:

- Supervised Learning (failure prediction)
- Unsupervised Learning (pattern discovery)
- Association Rule Mining (knowledge extraction)

All implemented using **WEKA Java API**.

---

# 10. Expected Learning Outcome

After implementation the project should illustrate:

- Predictive maintenance using machine learning
- Pattern discovery in engine telemetry data
- Knowledge extraction through association rules
- Practical use of the WEKA Java API in Java applications

