package qrels;

public interface EvalMetricTieBreaker {
    double[][] transform(double[] values);
}
