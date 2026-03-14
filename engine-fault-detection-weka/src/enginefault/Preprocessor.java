package enginefault;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.util.ArrayList;
import java.util.List;

/**
 * Preprocessor — cleans and transforms the raw engine fault detection dataset
 * for machine learning.
 *
 * Preprocessing Pipeline:
 *   1. Replace Missing Values
 *   2. Convert String to Nominal
 *   3. Convert Numeric Attributes to Nominal Levels
 *   4. Set Class Attribute (engine_condition)
 *   5. Balance Class Distribution (Resample with bias to oversample minority classes)
 *
 * Dataset attributes (from dataset-details.txt):
 *   Vibration_Amplitude  (0.1-10.0 mm/s²)   → vibration_amplitude_level  (low / medium / high)
 *   RMS_Vibration        (0.05-5.0 mm/s²)    → rms_vibration_level       (low / medium / high)
 *   Vibration_Frequency  (20-2000 Hz)         → vibration_frequency_level (low / medium / high)
 *   Surface_Temperature  (30-150 °C)          → surface_temperature_level (low / medium / high)
 *   Exhaust_Temperature  (200-600 °C)         → exhaust_temperature_level (low / medium / high)
 *   Acoustic_dB          (60-120 dB)          → acoustic_db_level         (low / medium / high)
 *   Acoustic_Frequency   (100-5000 Hz)        → acoustic_frequency_level  (low / medium / high)
 *   Intake_Pressure      (90-120 kPa)         → intake_pressure_level     (low / medium / high)
 *   Exhaust_Pressure     (80-110 kPa)         → exhaust_pressure_level    (low / medium / high)
 *   Frequency_Band_Energy (0.1-1.0)           → frequency_band_energy_level (low / medium / high)
 *   Amplitude_Mean       (0.01-0.5)           → amplitude_mean_level      (low / medium / high)
 *   Engine_Condition     (0/1/2)              → engine_condition (normal / minor_fault / critical_fault)
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

        // Step 4: Set Class Attribute to engine_condition
        int classIdx = -1;
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).name().equals("engine_condition")) {
                classIdx = i;
                break;
            }
        }
        if (classIdx >= 0) {
            data.setClassIndex(classIdx);
            System.out.println("Step 4: Set class attribute to 'engine_condition' (index " + classIdx + ").");
        } else {
            System.out.println("WARNING: engine_condition attribute not found!");
        }

        // Step 5: Balance Class Distribution using SpreadSubsample
        // Undersamples majority classes to match the minority class count
        System.out.println("\nBefore balancing — Class Distribution:");
        printClassDistribution(data);

        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0); // 1.0 = fully bias toward uniform class distribution
        resample.setNoReplacement(false);    // allow replacement (oversampling duplicates)
        // Target: each class ≈ maxClassCount → total ≈ numClasses * maxClassCount
        int numClasses = data.classAttribute().numValues();
        int maxClassCount = getMaxClassCount(data);
        double targetPercent = 100.0 * numClasses * maxClassCount / data.numInstances();
        resample.setSampleSizePercent(targetPercent);
        resample.setRandomSeed(42);
        resample.setInputFormat(data);
        data = Filter.useFilter(data, resample);
        System.out.println("Step 5: Balanced class distribution (Resample — oversampling minority classes).");

        System.out.println("\nAfter balancing — Class Distribution:");
        printClassDistribution(data);

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
            System.out.printf("  [%2d] %-35s (%s)", i, attr.name(), type);
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
                System.out.printf("  %-35s  min=%.4f  max=%.4f  mean=%.4f%n",
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

    /**
     * Returns the count of the largest class in the dataset.
     */
    private int getMaxClassCount(Instances data) {
        int numClasses = data.classAttribute().numValues();
        int max = 0;
        for (int v = 0; v < numClasses; v++) {
            int count = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                if ((int) data.instance(i).classValue() == v) count++;
            }
            if (count > max) max = count;
        }
        return max;
    }

    /**
     * Prints the class distribution of the dataset (used before/after balancing).
     */
    private void printClassDistribution(Instances data) {
        if (data.classIndex() < 0) return;
        Attribute classAttr = data.classAttribute();
        for (int v = 0; v < classAttr.numValues(); v++) {
            int count = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                if ((int) data.instance(i).classValue() == v) count++;
            }
            System.out.printf("  %-20s: %d (%.1f%%)%n",
                    classAttr.value(v), count, (100.0 * count / data.numInstances()));
        }
    }

    // =========================================================================
    // Numeric to Nominal Conversion
    // =========================================================================

    /**
     * Converts all numeric attributes to nominal categorical levels based on
     * the dataset ranges specified in dataset-details.txt.
     *
     * @param data dataset with numeric attributes
     * @return new Instances with all nominal attributes
     */
    private Instances convertNumericToNominal(Instances data) {
        ArrayList<Attribute> newAttributes = new ArrayList<>();

        // vibration_amplitude_level: 0.1-10.0 → low (<3.4), medium (3.4-6.7), high (>6.7)
        newAttributes.add(createNominalAttribute("vibration_amplitude_level", "low", "medium", "high"));

        // rms_vibration_level: 0.05-5.0 → low (<1.7), medium (1.7-3.4), high (>3.4)
        newAttributes.add(createNominalAttribute("rms_vibration_level", "low", "medium", "high"));

        // vibration_frequency_level: 20-2000 → low (<680), medium (680-1340), high (>1340)
        newAttributes.add(createNominalAttribute("vibration_frequency_level", "low", "medium", "high"));

        // surface_temperature_level: 30-150 → low (<70), medium (70-110), high (>110)
        newAttributes.add(createNominalAttribute("surface_temperature_level", "low", "medium", "high"));

        // exhaust_temperature_level: 200-600 → low (<335), medium (335-465), high (>465)
        newAttributes.add(createNominalAttribute("exhaust_temperature_level", "low", "medium", "high"));

        // acoustic_db_level: 60-120 → low (<80), medium (80-100), high (>100)
        newAttributes.add(createNominalAttribute("acoustic_db_level", "low", "medium", "high"));

        // acoustic_frequency_level: 100-5000 → low (<1730), medium (1730-3370), high (>3370)
        newAttributes.add(createNominalAttribute("acoustic_frequency_level", "low", "medium", "high"));

        // intake_pressure_level: 90-120 → low (<100), medium (100-110), high (>110)
        newAttributes.add(createNominalAttribute("intake_pressure_level", "low", "medium", "high"));

        // exhaust_pressure_level: 80-110 → low (<90), medium (90-100), high (>100)
        newAttributes.add(createNominalAttribute("exhaust_pressure_level", "low", "medium", "high"));

        // frequency_band_energy_level: 0.1-1.0 → low (<0.4), medium (0.4-0.7), high (>0.7)
        newAttributes.add(createNominalAttribute("frequency_band_energy_level", "low", "medium", "high"));

        // amplitude_mean_level: 0.01-0.5 → low (<0.17), medium (0.17-0.34), high (>0.34)
        newAttributes.add(createNominalAttribute("amplitude_mean_level", "low", "medium", "high"));

        // engine_condition: 0→normal, 1→minor_fault, 2→critical_fault
        newAttributes.add(createNominalAttribute("engine_condition", "normal", "minor_fault", "critical_fault"));

        // Create the new dataset structure
        Instances newData = new Instances("engine_fault_detection", newAttributes, data.numInstances());

        // Find source attribute indices
        int vibAmpIdx    = findAttributeIndex(data, "vibration_amplitude");
        int rmsVibIdx    = findAttributeIndex(data, "rms_vibration");
        int vibFreqIdx   = findAttributeIndex(data, "vibration_frequency");
        int surfTempIdx  = findAttributeIndex(data, "surface_temperature");
        int exhTempIdx   = findAttributeIndex(data, "exhaust_temperature");
        int acDbIdx      = findAttributeIndex(data, "acoustic_db");
        int acFreqIdx    = findAttributeIndex(data, "acoustic_frequency");
        int intPressIdx  = findAttributeIndex(data, "intake_pressure");
        int exhPressIdx  = findAttributeIndex(data, "exhaust_pressure");
        int freqBandIdx  = findAttributeIndex(data, "frequency_band_energy");
        int ampMeanIdx   = findAttributeIndex(data, "amplitude_mean");
        int engCondIdx   = findAttributeIndex(data, "engine_condition");

        // Transform each instance
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] vals = new double[newAttributes.size()];
            int pos = 0;

            vals[pos++] = categorizeVibrationAmplitude(inst.value(vibAmpIdx));
            vals[pos++] = categorizeRMSVibration(inst.value(rmsVibIdx));
            vals[pos++] = categorizeVibrationFrequency(inst.value(vibFreqIdx));
            vals[pos++] = categorizeSurfaceTemperature(inst.value(surfTempIdx));
            vals[pos++] = categorizeExhaustTemperature(inst.value(exhTempIdx));
            vals[pos++] = categorizeAcousticDB(inst.value(acDbIdx));
            vals[pos++] = categorizeAcousticFrequency(inst.value(acFreqIdx));
            vals[pos++] = categorizeIntakePressure(inst.value(intPressIdx));
            vals[pos++] = categorizeExhaustPressure(inst.value(exhPressIdx));
            vals[pos++] = categorizeFrequencyBandEnergy(inst.value(freqBandIdx));
            vals[pos++] = categorizeAmplitudeMean(inst.value(ampMeanIdx));
            vals[pos++] = categorizeEngineCondition(inst.value(engCondIdx));

            newData.add(new DenseInstance(1.0, vals));
        }

        return newData;
    }

    // =========================================================================
    // Categorization Rules (based on dataset ranges from dataset-details.txt)
    // =========================================================================

    /** Vibration_Amplitude (0.1-10.0): <3.4 → low, 3.4-6.7 → medium, >6.7 → high */
    private double categorizeVibrationAmplitude(double value) {
        if (value < 3.4)       return 0; // low
        else if (value <= 6.7) return 1; // medium
        else                   return 2; // high
    }

    /** RMS_Vibration (0.05-5.0): <1.7 → low, 1.7-3.4 → medium, >3.4 → high */
    private double categorizeRMSVibration(double value) {
        if (value < 1.7)       return 0; // low
        else if (value <= 3.4) return 1; // medium
        else                   return 2; // high
    }

    /** Vibration_Frequency (20-2000): <680 → low, 680-1340 → medium, >1340 → high */
    private double categorizeVibrationFrequency(double value) {
        if (value < 680)        return 0; // low
        else if (value <= 1340) return 1; // medium
        else                    return 2; // high
    }

    /** Surface_Temperature (30-150): <70 → low, 70-110 → medium, >110 → high */
    private double categorizeSurfaceTemperature(double value) {
        if (value < 70)        return 0; // low
        else if (value <= 110) return 1; // medium
        else                   return 2; // high
    }

    /** Exhaust_Temperature (200-600): <335 → low, 335-465 → medium, >465 → high */
    private double categorizeExhaustTemperature(double value) {
        if (value < 335)       return 0; // low
        else if (value <= 465) return 1; // medium
        else                   return 2; // high
    }

    /** Acoustic_dB (60-120): <80 → low, 80-100 → medium, >100 → high */
    private double categorizeAcousticDB(double value) {
        if (value < 80)        return 0; // low
        else if (value <= 100) return 1; // medium
        else                   return 2; // high
    }

    /** Acoustic_Frequency (100-5000): <1730 → low, 1730-3370 → medium, >3370 → high */
    private double categorizeAcousticFrequency(double value) {
        if (value < 1730)       return 0; // low
        else if (value <= 3370) return 1; // medium
        else                    return 2; // high
    }

    /** Intake_Pressure (90-120): <100 → low, 100-110 → medium, >110 → high */
    private double categorizeIntakePressure(double value) {
        if (value < 100)       return 0; // low
        else if (value <= 110) return 1; // medium
        else                   return 2; // high
    }

    /** Exhaust_Pressure (80-110): <90 → low, 90-100 → medium, >100 → high */
    private double categorizeExhaustPressure(double value) {
        if (value < 90)        return 0; // low
        else if (value <= 100) return 1; // medium
        else                   return 2; // high
    }

    /** Frequency_Band_Energy (0.1-1.0): <0.4 → low, 0.4-0.7 → medium, >0.7 → high */
    private double categorizeFrequencyBandEnergy(double value) {
        if (value < 0.4)       return 0; // low
        else if (value <= 0.7) return 1; // medium
        else                   return 2; // high
    }

    /** Amplitude_Mean (0.01-0.5): <0.17 → low, 0.17-0.34 → medium, >0.34 → high */
    private double categorizeAmplitudeMean(double value) {
        if (value < 0.17)       return 0; // low
        else if (value <= 0.34) return 1; // medium
        else                    return 2; // high
    }

    /** Engine_Condition: 0 → normal, 1 → minor_fault, 2 → critical_fault */
    private double categorizeEngineCondition(double value) {
        int v = (int) Math.round(value);
        if (v <= 0)      return 0; // normal
        else if (v == 1) return 1; // minor_fault
        else             return 2; // critical_fault
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /** Creates a nominal WEKA attribute with the given name and values. */
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

