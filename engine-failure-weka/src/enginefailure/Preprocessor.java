package enginefailure;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.util.ArrayList;
import java.util.List;

/**
 * Preprocessor — cleans and transforms the raw dataset for machine learning.
 *
 * Preprocessing Pipeline (per spec):
 *   1. Replace Missing Values
 *   2. Convert String to Nominal
 *   3. Convert Numeric Attributes to Nominal Levels
 *   4. Set Class Attribute (fault_condition)
 *
 * Numeric-to-Nominal conversion rules:
 *   Temperature  → temperature_level   (low / medium / high)
 *   RPM          → rpm_level           (low / medium / high)
 *   Fuel_Efficiency → fuel_efficiency_level (good / moderate / poor)
 *   Torque       → torque_level        (low / medium / high)
 *   Power_Output → power_output_level  (low / medium / high)
 *   Fault_Condition (0-3) → fault_condition (normal / minor_fault / moderate_fault / severe_fault)
 *   Operational_Mode → kept as-is (already categorical)
 */
public class Preprocessor {

    /**
     * Applies the full preprocessing pipeline to the dataset.
     *
     * @param data raw Instances loaded from CSV
     * @return preprocessed Instances with all nominal attributes and class set
     * @throws Exception if any filter or transformation fails
     */
    public Instances preprocess(Instances data) throws Exception {
        System.out.println("\n=============================================");
        System.out.println("=== Preprocessing Dataset ===");
        System.out.println("=============================================");

        // Step 1: Replace Missing Values
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        data = Filter.useFilter(data, replaceMissing);
        System.out.println("Step 1: Replaced missing values.");

        // Step 2: Convert String to Nominal
        StringToNominal stringToNominal = new StringToNominal();
        stringToNominal.setAttributeRange("first-last");
        stringToNominal.setInputFormat(data);
        data = Filter.useFilter(data, stringToNominal);
        System.out.println("Step 2: Converted strings to nominal.");

        // Step 3: Convert Numeric Attributes to Nominal Levels
        data = convertNumericToNominal(data);
        System.out.println("Step 3: Converted numeric attributes to nominal levels.");

        // Step 4: Set Class Attribute to fault_condition
        int classIdx = -1;
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).name().equals("fault_condition")) {
                classIdx = i;
                break;
            }
        }
        if (classIdx >= 0) {
            data.setClassIndex(classIdx);
            System.out.println("Step 4: Set class attribute to 'fault_condition' (index " + classIdx + ").");
        } else {
            System.out.println("WARNING: fault_condition attribute not found!");
        }

        System.out.println("\nPreprocessing complete: " + data.numInstances() + " instances, "
                + data.numAttributes() + " attributes.");
        return data;
    }

    /**
     * Prints a summary of the dataset including attribute information and class distribution.
     *
     * @param data the Instances dataset to summarize
     */
    public void printSummary(Instances data) {
        System.out.println("\n=============================================");
        System.out.println("=== Dataset Summary ===");
        System.out.println("=============================================");
        System.out.println("Relation Name:       " + data.relationName());
        System.out.println("Number of Instances: " + data.numInstances());
        System.out.println("Number of Attributes:" + data.numAttributes());

        System.out.println("\nAttributes:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            String type = Attribute.typeToString(attr);
            System.out.printf("  [%2d] %-30s (%s)", i, attr.name(), type);
            if (attr.isNominal()) {
                System.out.print("  values: {");
                for (int v = 0; v < attr.numValues(); v++) {
                    if (v > 0) System.out.print(", ");
                    System.out.print(attr.value(v));
                }
                System.out.print("}");
            }
            System.out.println();
        }

        // Print class distribution if class is set
        if (data.classIndex() >= 0) {
            Attribute classAttr = data.classAttribute();
            System.out.println("\nClass Attribute: " + classAttr.name());
            System.out.println("Class Distribution:");
            for (int v = 0; v < classAttr.numValues(); v++) {
                int count = 0;
                for (int i = 0; i < data.numInstances(); i++) {
                    if ((int) data.instance(i).classValue() == v) {
                        count++;
                    }
                }
                System.out.printf("  %-20s: %d (%.1f%%)%n",
                        classAttr.value(v), count, (100.0 * count / data.numInstances()));
            }
        }

        // Print basic numeric statistics for numeric attributes
        System.out.println("\nNumeric Attribute Statistics:");
        boolean hasNumeric = false;
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                hasNumeric = true;
                System.out.printf("  %-30s  min=%.2f  max=%.2f  mean=%.2f%n",
                        data.attribute(i).name(),
                        data.attributeStats(i).numericStats.min,
                        data.attributeStats(i).numericStats.max,
                        data.attributeStats(i).numericStats.mean);
            }
        }
        if (!hasNumeric) {
            System.out.println("  (All attributes are nominal)");
        }
    }

    // =========================================================================
    // Numeric to Nominal Conversion
    // =========================================================================

    /**
     * Converts all numeric attributes to nominal categorical levels based on
     * the categorization rules defined in the project specification.
     *
     * @param data dataset after timestamp removal (has numeric + nominal attributes)
     * @return new Instances with all nominal attributes
     */
    private Instances convertNumericToNominal(Instances data) {
        // Build the list of new nominal attributes
        ArrayList<Attribute> newAttributes = new ArrayList<>();

        // temperature_level: low / medium / high
        newAttributes.add(createNominalAttribute("temperature_level", "low", "medium", "high"));

        // rpm_level: low / medium / high
        newAttributes.add(createNominalAttribute("rpm_level", "low", "medium", "high"));

        // fuel_efficiency_level: good / moderate / poor
        newAttributes.add(createNominalAttribute("fuel_efficiency_level", "good", "moderate", "poor"));

        // torque_level: low / medium / high
        newAttributes.add(createNominalAttribute("torque_level", "low", "medium", "high"));

        // power_output_level: low / medium / high
        newAttributes.add(createNominalAttribute("power_output_level", "low", "medium", "high"));

        // operational_mode — copy the existing nominal attribute values
        int opModeIdx = findAttributeIndex(data, "operational_mode");
        if (opModeIdx >= 0) {
            Attribute opMode = data.attribute(opModeIdx);
            List<String> opValues = new ArrayList<>();
            for (int v = 0; v < opMode.numValues(); v++) {
                opValues.add(opMode.value(v));
            }
            newAttributes.add(new Attribute("operational_mode", opValues));
        }

        // fault_condition: normal / minor_fault / moderate_fault / severe_fault
        newAttributes.add(createNominalAttribute("fault_condition",
                "normal", "minor_fault", "moderate_fault", "severe_fault"));

        // Create the new dataset structure
        Instances newData = new Instances("engine_failure", newAttributes, data.numInstances());

        // Find source attribute indices (in the original data after timestamp removal)
        int tempIdx   = findAttributeIndex(data, "temperature");
        int rpmIdx    = findAttributeIndex(data, "rpm");
        int fuelIdx   = findAttributeIndex(data, "fuel_efficiency");
        int torqueIdx = findAttributeIndex(data, "torque");
        int powerIdx  = findAttributeIndex(data, "power_output");
        int faultIdx  = findAttributeIndex(data, "fault_condition");

        // Transform each instance
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] vals = new double[newAttributes.size()];
            int attrPos = 0;

            // Temperature → temperature_level
            vals[attrPos++] = categorizeTemperature(inst.value(tempIdx));

            // RPM → rpm_level
            vals[attrPos++] = categorizeRPM(inst.value(rpmIdx));

            // Fuel Efficiency → fuel_efficiency_level
            vals[attrPos++] = categorizeFuelEfficiency(inst.value(fuelIdx));

            // Torque → torque_level
            vals[attrPos++] = categorizeTorque(inst.value(torqueIdx));

            // Power Output → power_output_level
            vals[attrPos++] = categorizePowerOutput(inst.value(powerIdx));

            // Operational Mode — copy nominal index directly
            if (opModeIdx >= 0) {
                vals[attrPos++] = inst.value(opModeIdx);
            }

            // Fault Condition → fault_condition (numeric 0-3 → named categories)
            vals[attrPos++] = categorizeFaultCondition(inst.value(faultIdx));

            newData.add(new DenseInstance(1.0, vals));
        }

        return newData;
    }

    // =========================================================================
    // Categorization Rules (from project specification)
    // =========================================================================

    /**
     * Temperature: < 70 → low, 70–90 → medium, > 90 → high
     */
    private double categorizeTemperature(double value) {
        if (value < 70)       return 0; // low
        else if (value <= 90) return 1; // medium
        else                  return 2; // high
    }

    /**
     * RPM: < 2000 → low, 2000–4500 → medium, > 4500 → high
     */
    private double categorizeRPM(double value) {
        if (value < 2000)       return 0; // low
        else if (value <= 3000) return 1; // medium
        else                    return 2; // high
    }

    /**
     * Fuel Efficiency: > 18 → good, 12–18 → moderate, < 12 → poor
     */
    private double categorizeFuelEfficiency(double value) {
        if (value > 25)       return 0; // good
        else if (value >= 20) return 1; // moderate
        else                  return 2; // poor
    }

    /**
     * Torque: < 120 → low, 120–200 → medium, > 200 → high
     */
    private double categorizeTorque(double value) {
        if (value < 100)       return 0; // low
        else if (value <= 150) return 1; // medium
        else                   return 2; // high
    }

    /**
     * Power Output: < 60 → low, 60–120 → medium, > 120 → high
     */
    private double categorizePowerOutput(double value) {
        if (value < 40)        return 0; // low
        else if (value <= 80) return 1; // medium
        else                   return 2; // high
    }

    /**
     * Fault Condition: 0 → normal, 1 → minor_fault, 2 → moderate_fault, 3 → severe_fault
     */
    private double categorizeFaultCondition(double value) {
        int v = (int) Math.round(value);
        if (v <= 0)     return 0; // normal
        else if (v == 1) return 1; // minor_fault
        else if (v == 2) return 2; // moderate_fault
        else             return 3; // severe_fault
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /**
     * Creates a nominal WEKA attribute with the given name and values.
     */
    private Attribute createNominalAttribute(String name, String... values) {
        List<String> vals = new ArrayList<>();
        for (String v : values) {
            vals.add(v);
        }
        return new Attribute(name, vals);
    }

    /**
     * Finds an attribute index by name using case-insensitive partial matching.
     * Returns -1 if not found.
     *
     * @param data  the dataset to search
     * @param names one or more name fragments to match
     * @return the index of the first matching attribute, or -1
     */
    private int findAttributeIndex(Instances data, String... names) {
        for (int i = 0; i < data.numAttributes(); i++) {
            String attrName = data.attribute(i).name().toLowerCase();
            for (String name : names) {
                if (attrName.contains(name.toLowerCase())) {
                    return i;
                }
            }
        }
        return -1;
    }
}
