# CURSOR_ENGINE_FAILURE_PROJECT_SPEC.md

# Engine Failure Detection using Data Mining Techniques with WEKA

## Implementation Specification for Cursor

This document defines the exact implementation requirements for building
the Engine Failure Detection project using the WEKA Java API.

The instructions below override parts of the existing Developer Guide
where necessary, especially for class definition and numeric-to-nominal
conversion.

------------------------------------------------------------------------

# 1. Required Changes from Original Guide

## Class Attribute

The dataset originally contains:

fault_severity

This must be renamed and treated as the class attribute:

fault_condition

Possible values:

-   normal
-   minor_fault
-   moderate_fault
-   severe_fault

Implementation requirement:

dataset.setClass(dataset.attribute("fault_condition"));

------------------------------------------------------------------------

# 2. Numeric to Nominal Conversion

Important numeric attributes must be converted into categorical levels.

This improves: - interpretability - association rule mining - pattern
discovery

The conversion should be implemented in Preprocessor.java.

------------------------------------------------------------------------

# 3. Attribute Categorization Rules

## Temperature

Convert to: temperature_level

  Range      Category
  ---------- ----------
  \< 70      low
  70 -- 90   medium
  \> 90      high

------------------------------------------------------------------------

## RPM

Convert to: rpm_level

  Range          Category
  -------------- ----------
  \< 2000        low
  2000 -- 4500   medium
  \> 4500        high

------------------------------------------------------------------------

## Fuel Efficiency

Convert to: fuel_efficiency_level

  Range      Category
  ---------- ----------
  \> 18      good
  12 -- 18   moderate
  \< 12      poor

------------------------------------------------------------------------

## Vibration (X,Y,Z)

Convert to:

-   vibration_x_level
-   vibration_y_level
-   vibration_z_level

  Range        Category
  ------------ ----------------
  \< 1.5       stable
  1.5 -- 3.5   moderate
  \> 3.5       high_vibration

------------------------------------------------------------------------

## Torque

Convert to: torque_level

  Range        Category
  ------------ ----------
  \< 120       low
  120 -- 200   medium
  \> 200       high

------------------------------------------------------------------------

## Power Output

Convert to: power_output_level

  Range       Category
  ----------- ----------
  \< 60       low
  60 -- 120   medium
  \> 120      high

------------------------------------------------------------------------

## Operational Mode

Already categorical.

Values: - idle - cruising - heavy_load

No transformation required.

------------------------------------------------------------------------

## Timestamp

Timestamp must NOT be used for modeling.

Remove this attribute during preprocessing.

Example:

Remove remove = new Remove(); remove.setAttributeIndices("timestamp");

------------------------------------------------------------------------

# 4. Preprocessing Pipeline

The pipeline must run in this order:

Load CSV ↓ Replace Missing Values ↓ Convert String to Nominal ↓ Remove
Timestamp ↓ Convert Numeric Attributes → Nominal Levels ↓ Set Class
Attribute (fault_condition)

------------------------------------------------------------------------

# 5. Example Transformation

Original record:

temperature = 95\
rpm = 5100\
fuel_efficiency = 10\
vibration_x = 4.1\
torque = 230\
power_output = 140\
fault_condition = severe_fault

Converted record:

temperature_level = high\
rpm_level = high\
fuel_efficiency_level = poor\
vibration_x_level = high_vibration\
torque_level = high\
power_output_level = high\
fault_condition = severe_fault

------------------------------------------------------------------------

# 6. Association Rule Mining

Algorithm:

Apriori

Parameters:

minimum support = 0.1\
minimum confidence = 0.7

Example rules:

temperature_level=high ∧ vibration_x_level=high_vibration →
fault_condition=severe_fault

rpm_level=high ∧ torque_level=high → fault_condition=moderate_fault

operational_mode=heavy_load ∧ temperature_level=high →
fault_condition=severe_fault

------------------------------------------------------------------------

# 7. Supervised Learning

Algorithms:

-   RandomForest
-   J48
-   SMO

Evaluation:

10-fold cross validation

Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC AUC
-   Confusion Matrix

------------------------------------------------------------------------

# 8. Unsupervised Learning

Algorithm:

SimpleKMeans

Clusters:

k = 4

Goal:

Identify clusters dominated by: - normal - minor_fault -
moderate_fault - severe_fault

------------------------------------------------------------------------

# 9. Model Persistence

Save best classifier to:

models/best_model.model

Using:

SerializationHelper.write()\
SerializationHelper.read()

------------------------------------------------------------------------

# 10. Expected Output

Program should print:

-   Dataset summary
-   Classification evaluation results
-   Confusion matrix
-   Cluster centroids
-   Association rules
-   Sample prediction

Example:

Predicted Fault Condition: moderate_fault
