package experiments;

import org.apache.lucene.search.TopDocs;
import qpp.*;
import qrels.*;
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
                               Evaluator evaluator,
                               double[] evaluatedMetricValues,
                               EvalMetricTieBreaker tieBreaker) {
        int i;
        Map<String, TopDocs> topDocsMap = evaluator.getAllRetrievedResults().castToTopDocs();
        double[][] evaluatedMetricMatrix = tieBreaker.transform(evaluatedMetricValues);

        double[] qppEstimates = new double[queries.size()];

        i = 0;
        for (MsMarcoQuery query: queries) {
            qppEstimates[i] = qppMethod.computeSpecificity(query, topDocsMap.get(query.getId()), 50);
            i++;
        }

        List<TauAndSARE> tauAndSAREList = new ArrayList<>();
        for (i=0; i < evaluatedMetricMatrix.length; i++) {
            TauAndSARE tauAndSARE = new TauAndSARE(evaluatedMetricMatrix[i], qppEstimates);
            tauAndSAREList.add(new TauAndSARE(evaluatedMetricMatrix[i], qppEstimates));
            System.out.println(evaluatedMetricMatrix[i]);
            System.out.println(tauAndSARE.tau());
        }

        double tau_mean = tauAndSAREList.stream()
                                   .map(x -> x.tau())
                                   .mapToDouble(Double::doubleValue)
                                   .average()
                                   .orElse(0.0);

        List<double[]> perQuerySAREValuesList =
                tauAndSAREList.stream().map(x->x.getPerQuerySARE()).collect(Collectors.toList());
        double[] mean_sare = new double[perQuerySAREValuesList.get(0).length];
        for (double[] perQuerySAREValues: perQuerySAREValuesList) {
            for (int j=0; j < perQuerySAREValues.length; j++) {
                mean_sare[j] += perQuerySAREValues[j];
            }
            mean_sare = Arrays.stream(mean_sare).map(x->x/perQuerySAREValues.length).toArray();
        }

        return new TauAndSARE(tau_mean, mean_sare);
    }

    static TauAndSARE evaluate(List<MsMarcoQuery> queries,
                         QPPMethod qppMethod,
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

    public static void runExperiment(EvalMetricTieBreaker evalMetricTieBreaker)
            throws Exception {
        List<MsMarcoQuery> queries;
        final String resFile =
                Constants.BM25_Top100_DL1920
                //Constants.ColBERT_Top100_DL1920
                ;
        Metric targetMetric = Metric.P_10;
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
                new OddsRatioSpecificity(retriever.getSearcher(), 0.4f),
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
        List<QPPMethod> evaluatedQPPModels = new ArrayList<>();

        double[] evaluatedMetricValues = new double[queries.size()];
        int i=0;
        for (MsMarcoQuery query: queries) {
            evaluatedMetricValues[i++] = evaluator.compute(query.getId(), targetMetric);
        }
        //System.out.println(frequencyMap(Arrays.stream(evaluatedMetricValues).boxed()));

        for (QPPMethod qppMethod: qppMethods) {
            TauAndSARE qppMetrics = evaluate(
                    queries, qppMethod,
                    evaluator, evaluatedMetricValues,
                    evalMetricTieBreaker
            );
            System.out.println(String.format("%s on %s: tau = %.4f, sare = %.4f",
                    qppMethod.name(),
                    targetMetric.name(),
                    qppMetrics.tau(), qppMetrics.sare()));

            qppMethod.setMeasure(qppMetrics);
            evaluatedQPPModels.add(qppMethod);
        }

        List<QPPMethod> modelsRankedByPerfMeasure =
                evaluatedQPPModels.stream().sorted(
                                (o1, o2)->
                                        Double.compare(o1.getMeasure().tau(), o2.getMeasure().tau())
                        )
                        .collect(Collectors.toList());

        for (QPPMethod qppModel: modelsRankedByPerfMeasure) {
            System.out.println(String.format("%s: %.4f", qppModel.name(), qppModel.getMeasure().tau()));
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Evaluation w/o breaking ties");
        runExperiment(new NoTieBreaker());
        System.out.println("Evaluation w/ tie resolution over gropus (aggregate over max " +
                PermAggrTieBreaker.MAX_PERM + " permutations)");
        runExperiment(new PermAggrTieBreaker());
    }
}
