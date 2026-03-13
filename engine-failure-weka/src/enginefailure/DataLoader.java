package enginefailure;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;

/**
 * DataLoader — responsible for loading CSV datasets and converting them to ARFF format.
 *
 * Uses WEKA's CSVLoader to read CSV files and ArffSaver to persist them in ARFF format,
 * which is WEKA's native dataset representation.
 */
public class DataLoader {

    /**
     * Loads a CSV file and returns it as a WEKA Instances object.
     *
     * @param csvPath path to the CSV file
     * @return Instances dataset loaded from the CSV
     * @throws IOException if the file cannot be read
     */
    public Instances loadCSV(String csvPath) throws IOException {
        System.out.println("=== Loading CSV Dataset ===");
        System.out.println("File: " + csvPath);

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvPath));
        Instances data = loader.getDataSet();

        System.out.println("Loaded " + data.numInstances() + " instances with "
                + data.numAttributes() + " attributes.");
        return data;
    }

    /**
     * Converts a WEKA Instances dataset to ARFF format and saves it to disk.
     *
     * @param data     the Instances dataset to save
     * @param arffPath the output ARFF file path
     * @throws IOException if the file cannot be written
     */
    public void saveAsARFF(Instances data, String arffPath) throws IOException {
        System.out.println("\n=== Converting to ARFF ===");
        System.out.println("Output: " + arffPath);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffPath));
        saver.writeBatch();

        System.out.println("ARFF file saved successfully.");
    }

    /**
     * Convenience method: loads CSV and converts to ARFF in one step.
     *
     * @param csvPath  path to the CSV input file
     * @param arffPath path to the ARFF output file
     * @return Instances dataset
     * @throws IOException if file I/O fails
     */
    public Instances loadAndConvert(String csvPath, String arffPath) throws IOException {
        Instances data = loadCSV(csvPath);
        saveAsARFF(data, arffPath);
        return data;
    }
}



