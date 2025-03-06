package correlation;

import java.util.Arrays;
import java.util.Comparator;

public class NDCGCorrelation implements QPPCorrelationMetric {

    public static double computeNDCG(double[] labels, double[] scores) {
        if (labels.length != scores.length || labels.length == 0) {
            throw new IllegalArgumentException("Labels and scores must have the same length and be non-empty.");
        }

        int n = labels.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // Sort indices based on scores in descending order
        Arrays.sort(indices, Comparator.comparingDouble(i -> -scores[i]));

        double dcg = computeDCG(labels, indices);

        // Sort indices based on actual labels in descending order for ideal DCG
        Arrays.sort(indices, Comparator.comparingDouble(i -> -labels[i]));

        double idcg = computeDCG(labels, indices);

        return idcg == 0 ? 0 : dcg / idcg;
    }

    private static double computeDCG(double[] labels, Integer[] indices) {
        double dcg = 0.0;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            dcg += (Math.pow(2, labels[idx]) - 1) / (Math.log(i + 2) / Math.log(2));
        }
        return dcg;
    }

    public static void main(String[] args) {
        double[] labels = {2, 3, 0, 1, 2, 3};
        double[] scores = {0.8, 0.7, 0.6, 0.5, 0.4, 0.9};

        double ndcg = computeNDCG(labels, scores);
    }

    @Override
    public double correlation(double[] gt, double[] pred) {
        return computeNDCG(gt, pred);
    }

    @Override
    public String name() {
        return "ndcg";
    }
}
