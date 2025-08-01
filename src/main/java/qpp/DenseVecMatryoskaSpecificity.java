package qpp;

import org.apache.lucene.search.TopDocs;
import retrieval.MsMarcoQuery;

import java.util.List;
import java.util.Map;

public class DenseVecMatryoskaSpecificity extends DenseVecSpecificity {

    public DenseVecMatryoskaSpecificity(DocVectorReader vecReader, Map<Integer, float[]> queryVecs) {
        super(vecReader, queryVecs);
    }

    @Override
    public float computeDiameter(List<float[]> vecs, int k) {
        float weightedSum = 0.0f;

        for (int i = 1; i <= k; i++) {
            float diameter = super.computeDiameter(vecs, i);
            float weight = (float) (1.0 / Math.log(1 + i));  // i documents ⇒ log(1 + i)
            weightedSum += weight * diameter;
        }
        return weightedSum;
    }
}