package enginefault;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Main — entry point and pipeline orchestrator for the Engine Fault Detection system.
 *
 * Executes the complete data mining pipeline:
 *   1.  Load CSV dataset
 *   2.  Convert to ARFF
 *   3.  Print raw data summary
 *   4.  Preprocess dataset (ReplaceMissing → StringToNominal → NumericToNominal)
 *   5.  Print processed data summary
 *   6.  Supervised learning (RandomForest, J48, SMO — 10-fold CV)
 *   7.  K-Means clustering (k=3, matching the 3 engine condition classes)
 *   8.  Association rule mining (Apriori)
 *   9.  Save best model
 *   10. Load model
 *   11. Run sample prediction
 */
public class Main {

    /** Path to the raw CSV dataset */
    private static final String CSV_PATH = "data/engine_fault_detection_dataset.csv";

    /** Path to the ARFF output file */
    private static final String ARFF_PATH = "data/engine_fault_detection_dataset.arff";

    /** Path to save/load the best trained model */
    private static final String MODEL_PATH = "models/best_model.model";

    public static void main(String[] args) {
        try {
            System.out.println("=============================================");
            System.out.println("  Engine Fault Detection System");
            System.out.println("  Data Mining with WEKA");
            System.out.println("=============================================");

            // ----------------------------------------------------------
            // Steps 1-2: Load CSV and convert to ARFF
            // ----------------------------------------------------------
            DataLoader loader = new DataLoader();
            Instances rawData = loader.loadAndConvert(CSV_PATH, ARFF_PATH);

            // ----------------------------------------------------------
            // Step 3: Print raw data summary
            // ----------------------------------------------------------
            Preprocessor preprocessor = new Preprocessor();
            preprocessor.printSummary(rawData);

            // ----------------------------------------------------------
            // Step 4: Preprocess dataset
            // ----------------------------------------------------------
            Instances processedData = preprocessor.preprocess(rawData);

            // ----------------------------------------------------------
            // Step 5: Print processed data summary
            // ----------------------------------------------------------
            preprocessor.printSummary(processedData);

            // ----------------------------------------------------------
            // Step 6: Supervised learning — train & evaluate classifiers
            // ----------------------------------------------------------
            SupervisedTrainer trainer = new SupervisedTrainer();
            trainer.trainAndEvaluate(processedData);

            // ----------------------------------------------------------
            // Step 7: Unsupervised learning — K-Means clustering (k=3)
            // ----------------------------------------------------------
            ClusterAnalyzer clusterAnalyzer = new ClusterAnalyzer(3);
            clusterAnalyzer.analyze(processedData);

            // ----------------------------------------------------------
            // Step 8: Association rule mining — Apriori
            // ----------------------------------------------------------
            AssociationMiner miner = new AssociationMiner(0.1, 0.6);
            miner.mine(processedData);

            // ----------------------------------------------------------
            // Step 9: Save best model to disk
            // ----------------------------------------------------------
            ModelPersistence persistence = new ModelPersistence();
            Classifier bestModel = trainer.getBestClassifier();
            persistence.saveModel(bestModel, MODEL_PATH);

            // ----------------------------------------------------------
            // Step 10: Load model from disk
            // ----------------------------------------------------------
            Classifier loadedModel = persistence.loadModel(MODEL_PATH);

            // ----------------------------------------------------------
            // Step 11: Run a sample prediction
            // ----------------------------------------------------------
            runSamplePrediction(loadedModel, processedData);

            System.out.println("\n=============================================");
            System.out.println("  Pipeline Complete!");
            System.out.println("=============================================");

        } catch (Exception e) {
            System.err.println("Error during pipeline execution: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Runs a sample prediction using the loaded model.
     * Creates a synthetic instance representing a high-stress engine scenario:
     *   All sensor levels set to "high" to simulate a critical fault scenario.
     *
     * @param model      the trained classifier
     * @param dataFormat the processed Instances (used for attribute structure)
     * @throws Exception if prediction fails
     */
    private static void runSamplePrediction(Classifier model, Instances dataFormat) throws Exception {
        System.out.println("\n=============================================");
        System.out.println("=== Sample Prediction ===");
        System.out.println("=============================================");

        // Create a new instance matching the processed dataset structure
        Instance sample = new DenseInstance(dataFormat.numAttributes());
        sample.setDataset(dataFormat);

        // Set attribute values for a high-stress scenario
        sample.setValue(dataFormat.attribute("vibration_amplitude_level"), "high");
        sample.setValue(dataFormat.attribute("rms_vibration_level"), "high");
        sample.setValue(dataFormat.attribute("vibration_frequency_level"), "high");
        sample.setValue(dataFormat.attribute("surface_temperature_level"), "high");
        sample.setValue(dataFormat.attribute("exhaust_temperature_level"), "high");
        sample.setValue(dataFormat.attribute("acoustic_db_level"), "high");
        sample.setValue(dataFormat.attribute("acoustic_frequency_level"), "high");
        sample.setValue(dataFormat.attribute("intake_pressure_level"), "high");
        sample.setValue(dataFormat.attribute("exhaust_pressure_level"), "high");
        sample.setValue(dataFormat.attribute("frequency_band_energy_level"), "high");
        sample.setValue(dataFormat.attribute("amplitude_mean_level"), "high");

        // Print the sample instance
        System.out.println("\nInput Instance:");
        for (int i = 0; i < dataFormat.numAttributes(); i++) {
            if (i == dataFormat.classIndex()) continue;
            System.out.println("  " + dataFormat.attribute(i).name() + " = " + sample.stringValue(i));
        }

        // Predict
        double prediction = model.classifyInstance(sample);
        String predictedClass = dataFormat.classAttribute().value((int) prediction);

        System.out.println("\nPredicted Engine Condition: " + predictedClass);

        // Print probability distribution
        double[] distribution = model.distributionForInstance(sample);
        System.out.println("\nClass Probability Distribution:");
        for (int c = 0; c < distribution.length; c++) {
            System.out.printf("  %-20s: %.4f (%.1f%%)%n",
                    dataFormat.classAttribute().value(c),
                    distribution[c],
                    distribution[c] * 100);
        }
    }
}

