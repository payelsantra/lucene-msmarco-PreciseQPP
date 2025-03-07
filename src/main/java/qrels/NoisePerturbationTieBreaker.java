package qrels;

import org.apache.commons.lang3.ArrayUtils;

public class NoisePerturbationTieBreaker implements EvalMetricTieBreaker {
    int numRankings;
    double delta;

    public NoisePerturbationTieBreaker(int numRankings, double delta) { this.numRankings = numRankings; this.delta = delta;}

    @Override
    public String name() {
        return "tau-aggr";
    }

    @Override
    public double[][] transform(double[] values) {
        double[][] noise = new double[numRankings][values.length];
        for (int i = 0; i < numRankings; i++) {
            for (int j = 0; j < values.length; j++) {
                double epsilon = -delta + 2*delta*Math.random();
                noise[i][j] = values[j] + epsilon;
            }
        }
        return noise;
    }

    public static void main(String[] args) {
        double[] values = {4.2, 2.1, 4.2, 3.3, 2.1, 3.3, 4.2, 4.2, 1.0, 2.1};
        double[][] tieResolvedValues = new NoisePerturbationTieBreaker(5, .01).transform(values);
        for (int i=0; i < tieResolvedValues.length; i++) {
            System.out.println(ArrayUtils.toString(tieResolvedValues[i]));
        }
    }
}
