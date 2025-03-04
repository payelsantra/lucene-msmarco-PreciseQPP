package experiments;

import org.apache.lucene.search.TopDocs;
import qpp.*;
import qrels.Evaluator;
import qrels.Metric;
import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;
import stochastic_qpp.TauAndSARE;
import utils.IndexUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.*;
import java.util.*;
import java.util.function.*;

public class QPPPrecHeavyEvaluator {

    static <T>Map<T, Long> frequencyMap(Stream<T> elements) {
        return elements.collect(
                Collectors.groupingBy(
                        Function.identity(),
                        HashMap::new, // can be skipped
                        Collectors.counting()
                )
        );
    }

    static TauAndSARE evaluate(List<MsMarcoQuery> queries,
                         QPPMethod qppMethod,
                         Metric targetMetric,
                         Evaluator evaluator, double[] evaluatedMetricValues) {
        int i;
        Map<String, TopDocs> topDocsMap = evaluator.getAllRetrievedResults().castToTopDocs();

        double[] qppEstimates = new double[queries.size()];

        i = 0;
        for (MsMarcoQuery query: queries) {
            qppEstimates[i] = qppMethod.computeSpecificity(query, topDocsMap.get(query.getId()), 50);
            /*
            System.out.println(String.format(
                    "%s = %.4f, %s = %.4f",
                    qppMethod.name(), qppEstimates[i], targetMetric.name(), evaluatedMetricValues[i]));
             */
            i++;
        }
        return new TauAndSARE(evaluatedMetricValues, qppEstimates);
    }

    public static void main(String[] args) throws Exception {
        List<MsMarcoQuery> queries;
        final String resFile =
                Constants.BM25_Top100_DL1920
                //Constants.ColBERT_Top100_DL1920
        ;
        final Metric[] targetMetricNames = {
                //Metric.nDCG_1,
                Metric.P_10,
                //Metric.RR
        };

        OneStepRetriever retriever = new OneStepRetriever(Constants.QUERIES_DL1920, resFile);

        QPPMethod[] qppMethods = {
                new NQCSpecificity(retriever.getSearcher()),
                new VariantSpecificity(
                    new NQCSpecificity(retriever.getSearcher()),
                    retriever.getSearcher(),
                    new KNNRelModel(Constants.QRELS_TRAIN, Constants.QUERIES_DL1920, false),
                    5, 0.5f
                ),
                new VariantSpecificity(
                        new NQCSpecificity(retriever.getSearcher()),
                        retriever.getSearcher(),
                        new KNNRelModel(Constants.QRELS_TRAIN, Constants.QUERIES_DL1920, false),
                        5, 0.2f
                ),
                new VariantSpecificity(
                        new NQCSpecificity(retriever.getSearcher()),
                        retriever.getSearcher(),
                        new KNNRelModel(Constants.QRELS_TRAIN, Constants.QUERIES_DL1920, false),
                        5, 0.8f
                ),
                new OddsRatioSpecificity(retriever.getSearcher(), 0.2f),
                new WIGSpecificity(retriever.getSearcher()),
                new NQCCalibratedSpecificity(retriever.getSearcher(), 0.33f, 0.33f, 0.33f),
                new NQCCalibratedSpecificity(retriever.getSearcher(), 0.8f, 0.1f, 0.1f),
                new NQCCalibratedSpecificity(retriever.getSearcher(), 0.1f, 0.8f, 0.1f),
                new NQCCalibratedSpecificity(retriever.getSearcher(), 0.1f, 0.1f, 0.8f),
                new RSDSpecificity(new NQCSpecificity(retriever.getSearcher())),
                new UEFSpecificity(new NQCSpecificity(retriever.getSearcher()))
        };

        queries = retriever.getQueryList();
        IndexUtils.init(retriever.getSearcher());

        Evaluator evaluator = new Evaluator(Constants.QRELS_DL1920, resFile, 10); // Metrics for top-10

        for (Metric targetMetric: targetMetricNames) {
            double[] evaluatedMetricValues = new double[queries.size()];
            int i=0;
            for (MsMarcoQuery query: queries) {
                evaluatedMetricValues[i++] = evaluator.compute(query.getId(), targetMetric);
            }
            System.out.println(frequencyMap(Arrays.stream(evaluatedMetricValues).boxed()));

            for (QPPMethod qppMethod: qppMethods) {
                TauAndSARE qppMetrics = evaluate(queries, qppMethod, targetMetric, evaluator, evaluatedMetricValues);
                System.out.println(String.format("%s on %s: tau = %.4f, sare = %.4f",
                        qppMethod.name(),
                        targetMetric.name(),
                        qppMetrics.tau(), qppMetrics.sare()));
            }
        }
    }
}
