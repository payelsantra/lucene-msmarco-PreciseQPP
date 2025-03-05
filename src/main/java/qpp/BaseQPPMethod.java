package qpp;

import org.apache.lucene.search.TopDocs;
import retrieval.MsMarcoQuery;
import stochastic_qpp.TauAndSARE;

import java.io.IOException;
import java.util.List;
import java.util.Map;

abstract public class BaseQPPMethod implements QPPMethod {
    TauAndSARE perf_measure;

    public void setDataSource(String dataFile) throws IOException {}
    public void writePermutationMap(
            List<MsMarcoQuery> queries,
            Map<String, TopDocs> topDocsMap, int sampleNumber) throws IOException {}

    @Override
    public void setMeasure(TauAndSARE perf_measure) { this.perf_measure = perf_measure; }
    @Override
    public TauAndSARE getMeasure() { return perf_measure; }
}
