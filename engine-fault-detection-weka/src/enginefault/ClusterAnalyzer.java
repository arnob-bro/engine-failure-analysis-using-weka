package enginefault;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * ClusterAnalyzer — applies unsupervised learning (K-Means) to detect patterns
 * in engine sensor data.
 *
 * The class attribute is removed before clustering so the algorithm works
 * purely on the feature space. After clustering, results are cross-tabulated
 * against engine condition labels to interpret the clusters.
 */
public class ClusterAnalyzer {

    /** Number of clusters to form */
    private int numClusters;

    /** The K-Means clusterer instance */
    private SimpleKMeans kMeans;

    /**
     * Creates a ClusterAnalyzer with the specified number of clusters.
     *
     * @param numClusters number of clusters (e.g. 3)
     */
    public ClusterAnalyzer(int numClusters) {
        this.numClusters = numClusters;
    }

    /**
     * Performs K-Means clustering on the dataset.
     * The class attribute is removed before clustering.
     *
     * @param data preprocessed dataset with class attribute set
     * @throws Exception if clustering fails
     */
    public void analyze(Instances data) throws Exception {
        System.out.println("\n=============================================");
        System.out.println("=== Unsupervised Learning — K-Means Clustering ===");
        System.out.println("=============================================");

        // Remove the class attribute for clustering (unsupervised)
        Instances clusterData = removeClassAttribute(data);

        // Configure and run K-Means
        kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters);
        kMeans.setPreserveInstancesOrder(true);
        kMeans.setSeed(42);
        kMeans.buildClusterer(clusterData);

        // Print cluster centroids
        System.out.println("\nCluster Centroids:");
        System.out.println(kMeans.toString());

        // Print cluster assignments and cross-reference with engine condition
        printClusterAssignments(data, clusterData);
    }

    /**
     * Removes the class attribute from the dataset for unsupervised analysis.
     */
    private Instances removeClassAttribute(Instances data) throws Exception {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices(String.valueOf(data.classIndex() + 1));
        removeFilter.setInputFormat(data);
        return Filter.useFilter(data, removeFilter);
    }

    /**
     * Prints a cross-tabulation of clusters vs. engine condition classes.
     */
    private void printClusterAssignments(Instances originalData, Instances clusterData) throws Exception {
        int[] assignments = kMeans.getAssignments();

        int numClasses = originalData.classAttribute().numValues();
        int[][] crossTab = new int[numClusters][numClasses];

        for (int i = 0; i < assignments.length; i++) {
            int cluster = assignments[i];
            int classIdx = (int) originalData.instance(i).classValue();
            crossTab[cluster][classIdx]++;
        }

        // Print cross-tabulation
        System.out.println("\nCluster vs. Engine Condition Cross-Tabulation:");
        System.out.printf("%-10s", "Cluster");
        for (int c = 0; c < numClasses; c++) {
            System.out.printf("%-18s", originalData.classAttribute().value(c));
        }
        System.out.println("Total");
        System.out.println("-".repeat(10 + 18 * numClasses + 6));

        for (int k = 0; k < numClusters; k++) {
            System.out.printf("%-10s", "C" + k);
            int total = 0;
            for (int c = 0; c < numClasses; c++) {
                System.out.printf("%-18d", crossTab[k][c]);
                total += crossTab[k][c];
            }
            System.out.println(total);
        }

        // Interpretation
        System.out.println("\nInterpretation:");
        for (int k = 0; k < numClusters; k++) {
            int maxClassIdx = 0;
            int maxCount = crossTab[k][0];
            int total = 0;
            for (int c = 0; c < numClasses; c++) {
                total += crossTab[k][c];
                if (crossTab[k][c] > maxCount) {
                    maxCount = crossTab[k][c];
                    maxClassIdx = c;
                }
            }
            String dominantClass = originalData.classAttribute().value(maxClassIdx);
            double pct = (total > 0) ? (100.0 * maxCount / total) : 0;
            System.out.printf("  Cluster %d is predominantly '%s' (%.1f%% of %d instances)%n",
                    k, dominantClass, pct, total);
        }
    }
}

