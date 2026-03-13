# Developer Guide

## Engine Failure Detection using Data Mining Techniques with WEKA

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Environment Setup](#5-environment-setup)
6. [Build and Run Instructions](#6-build-and-run-instructions)
7. [Dataset Description](#7-dataset-description)
8. [Architecture and Module Reference](#8-architecture-and-module-reference)
   - 8.1 [DataLoader](#81-dataloader)
   - 8.2 [Preprocessor](#82-preprocessor)
   - 8.3 [SupervisedTrainer](#83-supervisedtrainer)
   - 8.4 [ClusterAnalyzer](#84-clusteranalyzer)
   - 8.5 [AssociationMiner](#85-associationminer)
   - 8.6 [ModelPersistence](#86-modelpersistence)
   - 8.7 [Main](#87-main-pipeline-orchestrator)
9. [Pipeline Execution Flow](#9-pipeline-execution-flow)
10. [WEKA API Reference](#10-weka-api-reference)
11. [Output Interpretation](#11-output-interpretation)
12. [Extending the Project](#12-extending-the-project)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

This project implements an **engine failure detection and analysis system** using multiple data mining techniques powered by the **WEKA Java API**. It processes synthetic engine telemetry data (sensor readings, operational modes, and fault severity labels) through a complete machine learning pipeline including:

- **Data Preprocessing** — cleaning, normalization, and format conversion
- **Supervised Learning** — multi-class classification to predict fault severity
- **Unsupervised Learning** — K-Means clustering to discover hidden patterns
- **Association Rule Mining** — Apriori algorithm to extract knowledge rules
- **Model Persistence** — saving and loading trained models for reuse

The system demonstrates how data mining can support **predictive maintenance** in industrial engine monitoring scenarios.

---

## 2. Technology Stack

| Component            | Technology                  |
|----------------------|-----------------------------|
| Language             | Java 11+                    |
| ML/DM Library        | WEKA Stable 3.8.6           |
| Build Tool           | Maven 3.x (or manual javac) |
| Dataset Format       | CSV (input) / ARFF (WEKA)   |
| IDE (recommended)    | IntelliJ IDEA / Eclipse     |

### Dependency JARs

The following JAR libraries are required at runtime (located in `lib/`):

| JAR File                            | Purpose                                  |
|-------------------------------------|------------------------------------------|
| `weka-stable-3.8.6.jar`            | Core WEKA machine learning library       |
| `bounce-0.18.jar`                   | Network authentication (WEKA dependency) |
| `java-cup-11b-20160615.jar`        | Parser generator (WEKA dependency)       |
| `java-cup-runtime-11b-20160615.jar`| CUP runtime (WEKA dependency)            |
| `commons-compress-1.21.jar`        | Compression utilities                    |
| `mtj-1.0.4.jar`                    | Matrix Toolkits for Java                 |
| `arpack_combined_all-0.1.jar`      | Eigenvalue computation                   |
| `netlib-java-1.1.jar`              | Numerical linear algebra                 |
| `istack-commons-runtime-3.0.12.jar`| XML stack commons                        |
| `jakarta.activation-api-1.2.2.jar` | Jakarta Activation API                   |
| `jakarta.xml.bind-api-2.3.3.jar`   | Jakarta XML Bind API                     |
| `jaxb-runtime-2.3.5.jar`           | JAXB Runtime                             |

---

## 3. Project Structure

```
engine-failure-weka/
│
├── data/
│   ├── engine_failure_dataset.csv        # Raw telemetry dataset (101 records)
│   └── engine_failure_dataset.arff       # Auto-generated ARFF version
│
├── models/
│   └── best_model.model                  # Serialized best classifier
│
├── lib/
│   ├── weka-stable-3.8.6.jar            # WEKA core library
│   ├── bounce-0.18.jar                   # WEKA transitive dependency
│   ├── java-cup-11b-20160615.jar
│   ├── java-cup-runtime-11b-20160615.jar
│   ├── commons-compress-1.21.jar
│   ├── mtj-1.0.4.jar
│   ├── arpack_combined_all-0.1.jar
│   ├── netlib-java-1.1.jar
│   ├── istack-commons-runtime-3.0.12.jar
│   ├── jakarta.activation-api-1.2.2.jar
│   ├── jakarta.xml.bind-api-2.3.3.jar
│   └── jaxb-runtime-2.3.5.jar
│
├── src/
│   └── enginefailure/
│       ├── Main.java                     # Pipeline orchestrator (entry point)
│       ├── DataLoader.java               # CSV loading and ARFF conversion
│       ├── Preprocessor.java             # Data cleaning and normalization
│       ├── SupervisedTrainer.java        # Classification (RF, J48, SMO)
│       ├── ClusterAnalyzer.java          # K-Means clustering
│       ├── AssociationMiner.java         # Apriori rule mining
│       └── ModelPersistence.java         # Model save/load utilities
│
├── out/                                  # Compiled .class files
├── pom.xml                               # Maven build configuration
└── DEVELOPER_GUIDE.md                    # This file
```

---

## 4. Prerequisites

Before building and running the project, ensure the following are installed:

1. **Java Development Kit (JDK) 11 or higher**
   - Verify: `java -version` and `javac -version`
   - Download: https://www.oracle.com/java/technologies/downloads/

2. **Maven 3.x** (optional — for Maven-based builds)
   - Verify: `mvn -version`
   - Download: https://maven.apache.org/download.cgi

> **Note:** Maven is optional. The project can be compiled and run directly with `javac`/`java` using the pre-downloaded JARs in the `lib/` folder.

---

## 5. Environment Setup

### Option A: Using Maven

If Maven is installed, dependencies are managed automatically via `pom.xml`:

```xml
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>weka-stable</artifactId>
    <version>3.8.6</version>
</dependency>
```

### Option B: Manual Setup (No Maven)

All required JARs are pre-downloaded in the `lib/` directory. No additional setup is needed — just ensure `javac` and `java` are available in your PATH.

### IDE Setup (IntelliJ IDEA)

1. Open IntelliJ IDEA and select **File > Open**, then choose the `engine-failure-weka` folder.
2. Mark `src/` as the **Sources Root** (right-click `src/` > **Mark Directory as > Sources Root**).
3. Add all JARs from `lib/` to the project classpath:
   - **File > Project Structure > Libraries > + > Java**
   - Select all `.jar` files inside `lib/`.
4. Set the **Run Configuration** main class to `enginefailure.Main`.
5. Set the **Working Directory** to the `engine-failure-weka/` root folder.

### IDE Setup (Eclipse)

1. Import the project as an **Existing Maven Project** (if using Maven) or a **Java Project**.
2. Right-click the project > **Build Path > Configure Build Path**.
3. Under the **Libraries** tab, click **Add External JARs** and select all JARs from `lib/`.
4. Create a Run Configuration with main class `enginefailure.Main`.
5. Set the working directory to the project root.

---

## 6. Build and Run Instructions

### Method 1: Using javac/java (Recommended)

**Compile:**

```powershell
cd engine-failure-weka
$jars = (Get-ChildItem lib\*.jar | ForEach-Object { $_.FullName }) -join ";"
javac -cp $jars -d out src\enginefailure\*.java
```

**Run:**

```powershell
java -cp "out;$jars" enginefailure.Main
```

> **Linux/macOS users:** Replace `;` with `:` in the classpath separator.

**Compile (Linux/macOS):**

```bash
cd engine-failure-weka
JARS=$(find lib -name "*.jar" | tr '\n' ':')
javac -cp "$JARS" -d out src/enginefailure/*.java
```

**Run (Linux/macOS):**

```bash
java -cp "out:$JARS" enginefailure.Main
```

### Method 2: Using Maven

```bash
cd engine-failure-weka
mvn compile
mvn exec:java -Dexec.mainClass="enginefailure.Main"
```

### Expected Runtime

The full pipeline typically completes in **10–30 seconds** depending on system performance. The majority of time is spent on the 10-fold cross-validation step in supervised learning.

---

## 7. Dataset Description

### Source File

`data/engine_failure_dataset.csv` — 101 synthetic engine telemetry records.

### Attributes (11 total)

| #  | Attribute          | Type    | Description                              | Range (approx.)       |
|----|--------------------|---------|------------------------------------------|-----------------------|
| 0  | `temperature`      | Numeric | Engine temperature (°C)                  | 68.2 – 128.0          |
| 1  | `rpm`              | Numeric | Revolutions per minute                   | 1000 – 5400           |
| 2  | `fuel_efficiency`  | Numeric | Fuel efficiency metric                   | 15.5 – 37.0           |
| 3  | `vibration_x`      | Numeric | Vibration on X-axis                      | 0.03 – 0.68           |
| 4  | `vibration_y`      | Numeric | Vibration on Y-axis                      | 0.02 – 0.60           |
| 5  | `vibration_z`      | Numeric | Vibration on Z-axis                      | 0.03 – 0.64           |
| 6  | `torque`           | Numeric | Engine torque (Nm)                       | 71.5 – 260.0          |
| 7  | `power_output`     | Numeric | Power output (kW)                        | 46.8 – 230.0          |
| 8  | `operational_mode` | Nominal | Engine operating mode                    | idle, cruising, heavy_load |
| 9  | `timestamp`        | Nominal | Datetime of the reading                  | 2024-01-01 08:00–16:20 |
| 10 | `fault_severity`   | Nominal | **Class label** — severity of fault      | normal, minor_fault, moderate_fault, severe_fault |

### Class Distribution

| Fault Severity   | Count | Percentage |
|------------------|-------|------------|
| normal           | 45    | 44.6%      |
| minor_fault      | 20    | 19.8%      |
| moderate_fault   | 18    | 17.8%      |
| severe_fault     | 18    | 17.8%      |

### Data Characteristics

- The dataset simulates real-world sensor behavior: **higher temperatures, RPM, vibration, and torque correlate with more severe faults**.
- **Idle mode** records are predominantly normal; **heavy_load** records exhibit more faults.
- The `fuel_efficiency` attribute has an **inverse relationship** with fault severity (lower efficiency indicates problems).

---

## 8. Architecture and Module Reference

The project follows a **modular architecture** where each Java class has a single, well-defined responsibility. All classes reside in the `enginefailure` package.

### Class Diagram

```
                    ┌──────────────┐
                    │   Main.java  │  ← Entry point & pipeline orchestrator
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
  │  DataLoader  │ │ Preprocessor │ │ ModelPersistence  │
  └──────────────┘ └──────────────┘ └──────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
  ┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐
  │SupervisedTrainer │ │ClusterAnalyzer │ │ AssociationMiner  │
  └──────────────────┘ └────────────────┘ └──────────────────┘
```

---

### 8.1 DataLoader

**File:** `src/enginefailure/DataLoader.java`

**Responsibility:** Load the raw CSV dataset and convert it to WEKA's native ARFF format.

**WEKA Classes Used:**

| Class                          | Purpose                        |
|--------------------------------|--------------------------------|
| `weka.core.converters.CSVLoader`  | Reads CSV files into Instances |
| `weka.core.converters.ArffSaver`  | Writes Instances to ARFF files |

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `loadCSV` | `Instances loadCSV(String csvPath)` | Loads a CSV file and returns a WEKA `Instances` object |
| `saveAsARFF` | `void saveAsARFF(Instances data, String arffPath)` | Saves an `Instances` object as an ARFF file |
| `loadAndConvert` | `Instances loadAndConvert(String csvPath, String arffPath)` | Convenience method — loads CSV and saves ARFF in one call |

**Usage Example:**

```java
DataLoader loader = new DataLoader();
Instances data = loader.loadAndConvert("data/engine_failure_dataset.csv",
                                       "data/engine_failure_dataset.arff");
data.setClassIndex(data.numAttributes() - 1);
```

---

### 8.2 Preprocessor

**File:** `src/enginefailure/Preprocessor.java`

**Responsibility:** Clean and transform the raw dataset to prepare it for machine learning.

**Preprocessing Pipeline (3 steps):**

| Step | Filter                     | Purpose                                              |
|------|----------------------------|------------------------------------------------------|
| 1    | `ReplaceMissingValues`     | Replaces missing values with attribute mean (numeric) or mode (nominal) |
| 2    | `StringToNominal`          | Converts any string-type attributes to nominal type  |
| 3    | `Normalize`                | Scales all numeric attributes to the [0, 1] range    |

**WEKA Classes Used:**

| Class                                                 | Purpose                      |
|-------------------------------------------------------|------------------------------|
| `weka.filters.unsupervised.attribute.ReplaceMissingValues` | Imputes missing data    |
| `weka.filters.unsupervised.attribute.StringToNominal`      | String → Nominal conversion |
| `weka.filters.unsupervised.attribute.Normalize`            | Min-max normalization   |
| `weka.filters.Filter`                                      | Static filter application utility |

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `preprocess` | `Instances preprocess(Instances data)` | Applies all three preprocessing filters sequentially |
| `printSummary` | `void printSummary(Instances data)` | Prints attribute names, types, class info, and summary statistics |

**Usage Example:**

```java
Preprocessor preprocessor = new Preprocessor();
preprocessor.printSummary(rawData);             // Display summary
Instances processed = preprocessor.preprocess(rawData);  // Clean data
processed.setClassIndex(processed.numAttributes() - 1);  // Re-set class index
```

**Important Note:** After preprocessing, always re-set the class index since filters may restructure the dataset.

---

### 8.3 SupervisedTrainer

**File:** `src/enginefailure/SupervisedTrainer.java`

**Responsibility:** Train multiple classification algorithms and evaluate them using stratified 10-fold cross-validation. Automatically selects the best model.

**Algorithms:**

| Algorithm       | WEKA Class                          | Description                     |
|-----------------|-------------------------------------|---------------------------------|
| RandomForest    | `weka.classifiers.trees.RandomForest` | Ensemble of 100 decision trees |
| J48             | `weka.classifiers.trees.J48`         | C4.5 decision tree algorithm   |
| SMO             | `weka.classifiers.functions.SMO`     | Support Vector Machine (SVM)   |

**Evaluation Metrics Reported:**

| Metric           | Description                                        |
|------------------|----------------------------------------------------|
| Accuracy         | Overall percentage of correctly classified instances |
| Precision (wtd)  | Weighted average precision across all classes       |
| Recall (wtd)     | Weighted average recall across all classes          |
| F1 Score (wtd)   | Weighted harmonic mean of precision and recall      |
| ROC AUC (wtd)    | Weighted area under the ROC curve                  |
| Confusion Matrix | Matrix showing actual vs. predicted class counts   |

**Evaluation Method:** Stratified 10-fold cross-validation using `weka.classifiers.Evaluation.crossValidateModel()` with seed 42 for reproducibility.

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `trainAndEvaluate` | `void trainAndEvaluate(Instances data)` | Trains all classifiers, evaluates, and selects the best one |
| `getBestClassifier` | `Classifier getBestClassifier()` | Returns the best classifier (trained on full dataset) |
| `getBestClassifierName` | `String getBestClassifierName()` | Returns the name of the best classifier |
| `getBestAccuracy` | `double getBestAccuracy()` | Returns the best accuracy percentage |

**Usage Example:**

```java
SupervisedTrainer trainer = new SupervisedTrainer();
trainer.trainAndEvaluate(processedData);

Classifier best = trainer.getBestClassifier();        // e.g., RandomForest
String name = trainer.getBestClassifierName();         // "RandomForest"
double accuracy = trainer.getBestAccuracy();           // 92.08
```

**Selection Logic:** The classifier with the highest cross-validation accuracy is selected as the best. It is then retrained on the full dataset before being returned (for model persistence).

---

### 8.4 ClusterAnalyzer

**File:** `src/enginefailure/ClusterAnalyzer.java`

**Responsibility:** Apply unsupervised K-Means clustering to discover patterns in engine telemetry data without using class labels.

**Algorithm:** `weka.clusterers.SimpleKMeans`

**Configuration:**

| Parameter           | Value | Description                          |
|---------------------|-------|--------------------------------------|
| Number of clusters  | 4     | Configurable via constructor         |
| Seed                | 42    | Ensures reproducible results         |
| Preserve order      | true  | Maintains instance ordering          |

**Key Design Decision:** The class attribute (`fault_severity`) is **removed** before clustering using a `weka.filters.unsupervised.attribute.Remove` filter. This ensures the clustering is truly unsupervised.

**Output Includes:**

- Cluster centroids (mean feature values per cluster)
- Cross-tabulation of clusters vs. fault severity classes
- Interpretation of which fault severity dominates each cluster

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `ClusterAnalyzer(int numClusters)` | Creates analyzer with specified cluster count |
| `analyze` | `void analyze(Instances data)` | Runs K-Means and prints results |

**Usage Example:**

```java
ClusterAnalyzer analyzer = new ClusterAnalyzer(4);
analyzer.analyze(processedData);
```

---

### 8.5 AssociationMiner

**File:** `src/enginefailure/AssociationMiner.java`

**Responsibility:** Discover association rules (relationships) between engine parameters using the Apriori algorithm.

**Algorithm:** `weka.associations.Apriori`

**Configuration:**

| Parameter        | Value | Description                                    |
|------------------|-------|------------------------------------------------|
| Min Support      | 0.1   | Minimum fraction of instances for an itemset   |
| Min Confidence   | 0.7   | Minimum confidence threshold for rules         |
| Number of Rules  | 20    | Maximum number of top rules to output          |

**Preprocessing:** Numeric attributes are **discretized** into bins using `weka.filters.unsupervised.attribute.Discretize` before running Apriori (since association rules require categorical data).

**Output Includes:**

- Generated association rules ranked by confidence
- Lift, leverage, and conviction values for each rule

**Example Rules Discovered:**

```
vibration_z='(-inf-0.1]'  ==>  fault_severity=normal     (conf: 1.0, lift: 2.24)
torque='(0.4-0.5]'        ==>  operational_mode=cruising  (conf: 1.0, lift: 2.73)
```

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `AssociationMiner(double minSupport, double minConfidence)` | Creates miner with thresholds |
| `mine` | `void mine(Instances data)` | Discretizes data and runs Apriori |

**Usage Example:**

```java
AssociationMiner miner = new AssociationMiner(0.1, 0.7);
miner.mine(processedData);
```

---

### 8.6 ModelPersistence

**File:** `src/enginefailure/ModelPersistence.java`

**Responsibility:** Save trained classifiers to disk and load them back for later predictions.

**WEKA Class Used:** `weka.core.SerializationHelper`

**Public Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `saveModel` | `void saveModel(Classifier classifier, String filePath)` | Serializes a trained classifier to a `.model` file |
| `loadModel` | `Classifier loadModel(String filePath)` | Deserializes a classifier from a `.model` file |

**Usage Example:**

```java
ModelPersistence persistence = new ModelPersistence();

// Save
persistence.saveModel(bestClassifier, "models/best_model.model");

// Load
Classifier loaded = persistence.loadModel("models/best_model.model");
```

**File Format:** The saved `.model` file is a standard Java serialized object. It can only be loaded with the same (or compatible) version of WEKA.

---

### 8.7 Main (Pipeline Orchestrator)

**File:** `src/enginefailure/Main.java`

**Responsibility:** Entry point of the application. Orchestrates the full data mining pipeline end-to-end.

**Constants:**

| Constant     | Value                                | Description          |
|--------------|--------------------------------------|----------------------|
| `CSV_PATH`   | `data/engine_failure_dataset.csv`    | Input CSV file       |
| `ARFF_PATH`  | `data/engine_failure_dataset.arff`   | Output ARFF file     |
| `MODEL_PATH` | `models/best_model.model`            | Saved model location |

**Pipeline Steps (executed in order):**

| Step | Action                        | Module Used          |
|------|-------------------------------|----------------------|
| 1    | Load CSV dataset              | `DataLoader`         |
| 2    | Convert CSV to ARFF           | `DataLoader`         |
| 3    | Print raw data summary        | `Preprocessor`       |
| 4    | Preprocess dataset            | `Preprocessor`       |
| 5    | Print processed data summary  | `Main` (inline)      |
| 6    | Train & evaluate classifiers  | `SupervisedTrainer`  |
| 7    | K-Means clustering            | `ClusterAnalyzer`    |
| 8    | Association rule mining       | `AssociationMiner`   |
| 9    | Save best model               | `ModelPersistence`   |
| 10   | Load model                    | `ModelPersistence`   |
| 11   | Run sample prediction         | `Main.runSamplePrediction()` |

**Sample Prediction:** The `runSamplePrediction()` method creates a synthetic high-temperature, heavy-load instance with normalized attribute values and predicts its fault severity. It also outputs the full class probability distribution.

---

## 9. Pipeline Execution Flow

```
┌─────────────────────────┐
│   Load CSV Dataset      │  DataLoader.loadCSV()
│   (101 instances, 11    │
│    attributes)          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Convert to ARFF       │  DataLoader.saveAsARFF()
│   (WEKA native format)  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Preprocess Dataset    │  Preprocessor.preprocess()
│   1. Replace missing    │   ├── ReplaceMissingValues
│   2. String → Nominal   │   ├── StringToNominal
│   3. Normalize [0,1]    │   └── Normalize
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Exploratory Summary   │  Preprocessor.printSummary()
│   (stats, attributes,   │
│    class distribution)  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Supervised Training   │  SupervisedTrainer.trainAndEvaluate()
│   ├── RandomForest      │   10-fold cross-validation
│   ├── J48 Decision Tree │   Accuracy, Precision, Recall,
│   └── SMO (SVM)         │   F1, ROC AUC, Confusion Matrix
│   → Select best model   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Clustering Analysis   │  ClusterAnalyzer.analyze()
│   K-Means (k=4)         │   Cluster centroids
│   Cross-tab vs. class   │   Cluster ↔ Severity mapping
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Association Mining    │  AssociationMiner.mine()
│   Apriori algorithm     │   Discretize → Find rules
│   support=0.1           │   Top 20 rules with lift
│   confidence=0.7        │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Save Best Model       │  ModelPersistence.saveModel()
│   → models/best_model   │   SerializationHelper.write()
│     .model              │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Load Model &          │  ModelPersistence.loadModel()
│   Sample Prediction     │  Main.runSamplePrediction()
│   → "moderate_fault"    │
└─────────────────────────┘
```

---

## 10. WEKA API Reference

This section lists the key WEKA classes used in the project grouped by category.

### Data Handling

| Class | Package | Usage |
|-------|---------|-------|
| `Instances` | `weka.core` | Core dataset container — holds all data instances and attribute definitions |
| `Instance` | `weka.core` | Represents a single data record |
| `DenseInstance` | `weka.core` | Concrete implementation of Instance for dense data |
| `CSVLoader` | `weka.core.converters` | Reads CSV files into Instances |
| `ArffSaver` | `weka.core.converters` | Writes Instances to ARFF files |

### Filters (Preprocessing)

| Class | Package | Usage |
|-------|---------|-------|
| `Filter` | `weka.filters` | Static utility to apply filter transformations |
| `ReplaceMissingValues` | `weka.filters.unsupervised.attribute` | Imputes missing values |
| `StringToNominal` | `weka.filters.unsupervised.attribute` | Converts string attributes to nominal |
| `Normalize` | `weka.filters.unsupervised.attribute` | Normalizes numeric attributes to [0,1] |
| `Discretize` | `weka.filters.unsupervised.attribute` | Bins numeric attributes into intervals |
| `Remove` | `weka.filters.unsupervised.attribute` | Removes specified attributes |

### Classifiers (Supervised Learning)

| Class | Package | Usage |
|-------|---------|-------|
| `Classifier` | `weka.classifiers` | Abstract base class for all classifiers |
| `Evaluation` | `weka.classifiers` | Model evaluation with cross-validation and metrics |
| `RandomForest` | `weka.classifiers.trees` | Ensemble of random decision trees |
| `J48` | `weka.classifiers.trees` | C4.5 decision tree algorithm |
| `SMO` | `weka.classifiers.functions` | Sequential Minimal Optimization (SVM) |

### Clusterers (Unsupervised Learning)

| Class | Package | Usage |
|-------|---------|-------|
| `SimpleKMeans` | `weka.clusterers` | K-Means clustering algorithm |

### Association Rules

| Class | Package | Usage |
|-------|---------|-------|
| `Apriori` | `weka.associations` | Apriori algorithm for frequent itemsets and association rules |

### Utilities

| Class | Package | Usage |
|-------|---------|-------|
| `SerializationHelper` | `weka.core` | Save/load Java objects (trained models) to/from disk |

---

## 11. Output Interpretation

### Supervised Learning Results

The cross-validation results include metrics for each classifier:

- **Accuracy > 90%** indicates strong predictive performance on this dataset.
- **Confusion Matrix** rows represent actual classes; columns represent predicted classes. Diagonal values are correct predictions.
- **ROC AUC close to 1.0** indicates excellent class discrimination.

**Best model selection:** The classifier with the highest accuracy is automatically chosen, retrained on the full dataset, and saved.

### Clustering Results

The cross-tabulation table maps each cluster to fault severity classes:

- Clusters that align closely with a single fault severity class validate that the features contain meaningful patterns.
- Example interpretation:
  - Cluster with high temperature/vibration centroids → maps to severe_fault
  - Cluster with low values and idle mode → maps to normal

### Association Rules

Rules are ranked by confidence (and secondarily by lift):

- **Confidence = 1.0** means the rule holds for every matching instance.
- **Lift > 1.0** means the rule represents a genuine association (not random).
- **Example:** `vibration_z='(-inf-0.1]' ==> fault_severity=normal (conf: 1.0, lift: 2.24)` means that whenever vibration_z is very low, the fault severity is always normal, and this pattern occurs 2.24x more than expected by chance.

### Sample Prediction

The prediction output shows:

- The predicted class (e.g., `moderate_fault`)
- The probability distribution across all four classes, showing model confidence.

---

## 12. Extending the Project

### Adding a New Classifier

1. Open `SupervisedTrainer.java`.
2. Import the new classifier class (e.g., `weka.classifiers.bayes.NaiveBayes`).
3. Add it to the `classifiers` map in the constructor:

```java
import weka.classifiers.bayes.NaiveBayes;

// In constructor:
NaiveBayes nb = new NaiveBayes();
classifiers.put("Naive Bayes", nb);
```

No other changes are needed — the evaluation loop handles all classifiers generically.

### Changing the Number of Clusters

In `Main.java`, modify the constructor argument:

```java
ClusterAnalyzer clusterAnalyzer = new ClusterAnalyzer(5);  // 5 clusters instead of 4
```

### Adjusting Association Rule Thresholds

In `Main.java`, modify the constructor arguments:

```java
AssociationMiner miner = new AssociationMiner(0.05, 0.8);  // lower support, higher confidence
```

### Using a Different Dataset

1. Place the new CSV file in `data/`.
2. Update `CSV_PATH` and `ARFF_PATH` constants in `Main.java`.
3. Ensure the last column is the class label, or adjust the `setClassIndex()` call.
4. If the dataset has different nominal attributes, the `StringToNominal` filter will handle them automatically.

### Making Predictions on New Data

After running the pipeline once, you can reuse the saved model:

```java
ModelPersistence persistence = new ModelPersistence();
Classifier model = persistence.loadModel("models/best_model.model");

// Create instance matching the preprocessed dataset structure
Instance newRecord = new DenseInstance(dataFormat.numAttributes());
newRecord.setDataset(dataFormat);
newRecord.setValue(0, 0.75);  // normalized temperature
// ... set other attributes ...

double prediction = model.classifyInstance(newRecord);
String label = dataFormat.classAttribute().value((int) prediction);
System.out.println("Predicted: " + label);
```

---

## 13. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `NoClassDefFoundError: org/bounce/net/DefaultAuthenticator` | Missing WEKA dependency JAR | Ensure `bounce-0.18.jar` is on the classpath |
| `ClassNotFoundException` for WEKA classes | Incomplete classpath | Add all JARs from `lib/` to the classpath |
| `FileNotFoundException` for CSV | Wrong working directory | Run the program from the `engine-failure-weka/` root directory |
| `ArffSaver` exception | Write permissions | Ensure the `data/` directory exists and is writable |
| `java.lang.OutOfMemoryError` | Large dataset with many classifiers | Increase heap: `java -Xmx2g -cp ...` |
| `Evaluation` returns NaN for metrics | Too few instances per class for 10-fold CV | Reduce folds (e.g., 5-fold) or add more data |
| Model file is incompatible | WEKA version mismatch | Retrain and save the model with the current WEKA version |

### MTJ Warning

You may see this warning during execution:

```
WARNING: core mtj jar files are not available as resources to this classloader
```

This is a **harmless warning** from the Matrix Toolkits for Java library. It does not affect functionality — WEKA falls back to its pure Java implementation for linear algebra operations.

### Encoding Issues

If Unicode box-drawing characters display as `?` in your console, set your terminal to UTF-8:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

---

*This developer guide was created for the Engine Failure Detection using Data Mining Techniques with WEKA project.*



