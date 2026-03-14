package enginefault;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

/**
 * ModelPersistence — handles saving and loading of trained WEKA models.
 *
 * Uses WEKA's SerializationHelper to serialize classifiers to disk
 * and deserialize them for later predictions.
 */
public class ModelPersistence {

    /**
     * Saves a trained classifier to a file.
     *
     * @param classifier the trained Classifier to save
     * @param filePath   the file path to save the model (e.g. "models/best_model.model")
     * @throws Exception if serialization fails
     */
    public void saveModel(Classifier classifier, String filePath) throws Exception {
        System.out.println("\n=== Saving Model ===");
        System.out.println("Path: " + filePath);

        SerializationHelper.write(filePath, classifier);

        System.out.println("Model saved successfully.");
    }

    /**
     * Loads a trained classifier from a file.
     *
     * @param filePath the file path of the saved model
     * @return the deserialized Classifier
     * @throws Exception if deserialization fails
     */
    public Classifier loadModel(String filePath) throws Exception {
        System.out.println("\n=== Loading Model ===");
        System.out.println("Path: " + filePath);

        Classifier classifier = (Classifier) SerializationHelper.read(filePath);

        System.out.println("Model loaded successfully.");
        System.out.println("Classifier: " + classifier.getClass().getSimpleName());
        return classifier;
    }
}

