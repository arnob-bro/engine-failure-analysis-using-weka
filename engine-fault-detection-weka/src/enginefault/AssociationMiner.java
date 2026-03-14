package enginefault;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 * AssociationMiner — discovers association rules between engine sensor parameters
 * using the Apriori algorithm.
 *
 * Configuration:
 *   - Minimum support:    0.1
 *   - Minimum confidence: 0.7
 *   - Maximum rules:      20
 *
 * Since the preprocessed data is already fully nominal (after numeric-to-nominal
 * conversion in Preprocessor), Apriori can run directly. If any numeric attributes
 * remain, they are discretized automatically before mining.
 */
public class AssociationMiner {

    /** Minimum support threshold (fraction of instances) */
    private double minSupport;

    /** Minimum confidence threshold for rules */
    private double minConfidence;

    /**
     * Creates an AssociationMiner with the specified thresholds.
     *
     * @param minSupport    minimum support (e.g. 0.1)
     * @param minConfidence minimum confidence (e.g. 0.7)
     */
    public AssociationMiner(double minSupport, double minConfidence) {
        this.minSupport = minSupport;
        this.minConfidence = minConfidence;
    }

    /**
     * Runs Apriori association rule mining on the dataset.
     * If numeric attributes exist, they are discretized first.
     * The class index is unset before mining so all attributes participate.
     *
     * @param data preprocessed (nominal) dataset
     * @throws Exception if mining fails
     */
    public void mine(Instances data) throws Exception {
        System.out.println("\n=============================================");
        System.out.println("=== Association Rule Mining — Apriori ===");
        System.out.println("=============================================");

        // Work on a copy so we don't modify the original dataset
        Instances miningData = new Instances(data);

        // Discretize any remaining numeric attributes (safety net)
        boolean hasNumeric = false;
        for (int i = 0; i < miningData.numAttributes(); i++) {
            if (miningData.attribute(i).isNumeric()) {
                hasNumeric = true;
                break;
            }
        }
        if (hasNumeric) {
            Discretize discretize = new Discretize();
            discretize.setInputFormat(miningData);
            miningData = Filter.useFilter(miningData, discretize);
            System.out.println("Discretized remaining numeric attributes for Apriori.");
        }

        // Unset class index for association mining (all attributes participate)
        miningData.setClassIndex(-1);

        // Configure Apriori
        Apriori apriori = new Apriori();
        apriori.setLowerBoundMinSupport(minSupport);
        apriori.setMinMetric(minConfidence);
        apriori.setNumRules(20);

        System.out.printf("Parameters: minSupport=%.2f, minConfidence=%.2f, maxRules=%d%n",
                minSupport, minConfidence, 20);

        // Build associations
        apriori.buildAssociations(miningData);

        // Print discovered rules
        System.out.println("\nDiscovered Association Rules:");
        System.out.println(apriori.toString());
    }
}

