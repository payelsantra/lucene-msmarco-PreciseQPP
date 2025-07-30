package qpp;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import retrieval.MsMarcoQuery;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class DenseVecSpecificity extends BaseIDFSpecificity {
    ChunkedMMapEmbeddingReader vecReader;
    Map<Long, double[]> queryVecs;

    public DenseVecSpecificity(ChunkedMMapEmbeddingReader vecReader, Map<Long, double[]> queryVecs) {
        this.vecReader = vecReader;
        this.queryVecs = queryVecs;
    }

    @Override
    public double computeSpecificity(MsMarcoQuery q, TopDocs topDocs, int cutoff) {
        List<double[]> vecs = new ArrayList<>();
        double[] qvec = queryVecs.get(Long.parseLong(q.getId()));
        vecs.add(qvec);

        int k = Math.min(cutoff, topDocs.scoreDocs.length);
        for (int i=0; i < k; i++) {
            double[] dvec = vecReader.getVector(topDocs.scoreDocs[i].doc);
            vecs.add(dvec);
        }

        return computeDiameter(vecs, k);
    }

    public double computeDiameter(List<double[]> vecs, int cutoff) {
        if (vecs == null || vecs.size() < cutoff + 1) { // cutoff: 1,2,...k
            throw new IllegalArgumentException("Insufficient vectors: need at least 'cutoff' docs and 1 query vector.");
        }

        int dim = vecs.get(0).length;
        double[] minVals = new double[dim];
        double[] maxVals = new double[dim];

        // Start with the query (first) vector
        double[] queryVec = vecs.get(0);
        System.arraycopy(queryVec, 0, minVals, 0, dim);
        System.arraycopy(queryVec, 0, maxVals, 0, dim);

        // Include first `cutoff` doc vectors
        for (int i = 0; i <= cutoff; i++) { // docs start from index 1 (0 is the query which we include in computation)
            double[] dvec = vecs.get(i);
            for (int d = 0; d < dim; d++) {
                minVals[d] = Math.min(minVals[d], dvec[d]);
                maxVals[d] = Math.max(maxVals[d], dvec[d]);
            }
        }

        // Compute the diameter
        double diameter = 0.0;
        for (int d = 0; d < dim; d++) {
            diameter += (maxVals[d] - minVals[d]);
        }

        return -1/Math.log(diameter);
    }
}
