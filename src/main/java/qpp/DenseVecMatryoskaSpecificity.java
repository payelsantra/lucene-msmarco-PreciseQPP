package qpp;

import org.apache.lucene.search.TopDocs;
import retrieval.MsMarcoQuery;

import java.util.List;
import java.util.Map;

public class DenseVecMatryoskaSpecificity extends DenseVecSpecificity {

    public DenseVecMatryoskaSpecificity(ChunkedMMapEmbeddingReader vecReader, Map<Long, double[]> queryVecs) {
        super(vecReader, queryVecs);
    }

    @Override
    public double computeDiameter(List<double[]> vecs, int k) {
        double weightedSum = 0.0;

        for (int i = 1; i <= k; i++) {
            double diameter = super.computeDiameter(vecs, i);
            double weight = 1.0 / Math.log(1 + i);  // i documents ⇒ log(1 + i)
            weightedSum += weight * diameter;
        }
        return weightedSum;
    }
}