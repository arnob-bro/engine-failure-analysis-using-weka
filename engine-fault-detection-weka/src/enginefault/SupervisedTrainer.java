package enginefault;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * SupervisedTrainer — trains multiple classification algorithms and evaluates
 * them using stratified 10-fold cross-validation.
 *
 * Algorithms:
 *   - RandomForest (ensemble of 100 decision trees)
 *   - J48 (C4.5 decision tree)
 *   - SMO (Support Vector Machine)
 *
 * Metrics reported per classifier:
 *   - Accuracy, Precision (weighted), Recall (weighted),
 *     F1 Score (weighted), ROC AUC (weighted), Confusion Matrix
 *
 * The classifier with the highest cross-validation accuracy is selected as the
 * best model, retrained on the full dataset, and made available via getBestClassifier().
 */
public class SupervisedTrainer {

    /** Map of classifier name → classifier instance */
    private Map<String, Classifier> classifiers;

    /** The best classifier after evaluation (retrained on full data) */
    private Classifier bestClassifier;

    /** Name of the best classifier */
    private String bestClassifierName;

    /** Best cross-validation accuracy percentage */
    private double bestAccuracy;

    /**
     * Initializes the trainer with RandomForest, J48, and SMO classifiers.
     */
    public SupervisedTrainer() {
        classifiers = new LinkedHashMap<>();

        // RandomForest — ensemble of 100 trees
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        classifiers.put("RandomForest", rf);

        // J48 — C4.5 decision tree
        J48 j48 = new J48();
        classifiers.put("J48", j48);

        // SMO — Support Vector Machine
        SMO smo = new SMO();
        classifiers.put("SMO", smo);
    }

    /**
     * Trains all classifiers with 10-fold cross-validation, prints evaluation
     * metrics and confusion matrices, and selects the best classifier.
     *
     * @param data preprocessed dataset with class attribute set
     * @throws Exception if training or evaluation fails
     */
    public void trainAndEvaluate(Instances data) throws Exception {
        System.out.println("\n=============================================");
        System.out.println("=== Supervised Learning — Classification ===");
        System.out.println("=============================================");

        bestAccuracy = -1;

        for (Map.Entry<String, Classifier> entry : classifiers.entrySet()) {
            String name = entry.getKey();
            Classifier classifier = entry.getValue();

            System.out.println("\n--- " + name + " ---");

            // Stratified 10-fold cross-validation with seed 42
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new Random(42));

            // Print evaluation metrics
            double accuracy = eval.pctCorrect();
            System.out.printf("Accuracy:          %.2f%%%n", accuracy);
            System.out.printf("Precision (wtd):   %.4f%n", eval.weightedPrecision());
            System.out.printf("Recall (wtd):      %.4f%n", eval.weightedRecall());
            System.out.printf("F1 Score (wtd):    %.4f%n", eval.weightedFMeasure());

            double rocAuc = eval.weightedAreaUnderROC();
            if (Double.isNaN(rocAuc)) {
                System.out.println("ROC AUC (wtd):     N/A");
            } else {
                System.out.printf("ROC AUC (wtd):     %.4f%n", rocAuc);
            }

            // Print confusion matrix
            System.out.println("\nConfusion Matrix:");
            double[][] cm = eval.confusionMatrix();

            // Header row
            System.out.printf("%-18s", "");
            for (int c = 0; c < data.classAttribute().numValues(); c++) {
                System.out.printf("%-18s", data.classAttribute().value(c));
            }
            System.out.println();
            System.out.println("-".repeat(18 + 18 * data.classAttribute().numValues()));

            // Data rows
            for (int r = 0; r < cm.length; r++) {
                System.out.printf("%-18s", data.classAttribute().value(r));
                for (int c = 0; c < cm[r].length; c++) {
                    System.out.printf("%-18.0f", cm[r][c]);
                }
                System.out.println();
            }

            // Track the best classifier by accuracy
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestClassifierName = name;
            }
        }

        // Retrain the best classifier on the full dataset
        System.out.println("\n=============================================");
        System.out.printf("Best Classifier: %s (Accuracy: %.2f%%)%n", bestClassifierName, bestAccuracy);
        System.out.println("Retraining " + bestClassifierName + " on full dataset...");

        bestClassifier = classifiers.get(bestClassifierName);
        bestClassifier.buildClassifier(data);
        System.out.println("Training complete.");
    }

    /** Returns the best trained classifier (retrained on full dataset). */
    public Classifier getBestClassifier() {
        return bestClassifier;
    }

    /** Returns the name of the best classifier. */
    public String getBestClassifierName() {
        return bestClassifierName;
    }

    /** Returns the best cross-validation accuracy percentage. */
    public double getBestAccuracy() {
        return bestAccuracy;
    }
}

