package experiments;

import correlation.NDCGCorrelation;
import correlation.QPPCorrelationMetric;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.lucene.search.TopDocs;
import qpp.*;
import qrels.*;
import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;
import stochastic_qpp.QPPMetricBundle;
import utils.IndexUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.*;
import java.util.*;
import java.util.function.*;

public class QPPPrecHeavyEvaluator {
    static final int TAU = 0;
    static final int NDCG = 1;
    static double DELTA = 0.01;
    static final int MAX_PERM = 5;
    static final int NUM_RANKINGS = 100;
    static final int NUM_DOCS = 10;

    static <T>Map<T, Long> frequencyMap(Stream<T> elements) {
        return elements.collect(
                Collectors.groupingBy(
                        Function.identity(),
                        HashMap::new, // can be skipped
                        Collectors.counting()
                )
        );
    }

    static QPPMetricBundle evaluate(List<MsMarcoQuery> queries,
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

        List<QPPMetricBundle> QPPMetricBundleList = new ArrayList<>();
        for (i=0; i < evaluatedMetricMatrix.length; i++) {
            QPPMetricBundle QPPMetricBundle = new QPPMetricBundle(evaluatedMetricMatrix[i], qppEstimates);
            QPPMetricBundleList.add(new QPPMetricBundle(evaluatedMetricMatrix[i], qppEstimates));
            //System.out.println(String.format("tau[%d] = %.4f", i, tauAndSARE.tau()));
            //System.out.println(ArrayUtils.toString(evaluatedMetricMatrix[i]));
        }

        double tau_mean = QPPMetricBundleList.stream()
                                   .map(x -> x.tau())
                                   .mapToDouble(Double::doubleValue)
                                   .average()
                                   .orElse(0.0);
        double ndcg_mean = QPPMetricBundleList.stream()
                .map(x -> x.tau())
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);

        List<double[]> perQuerySAREValuesList =
                QPPMetricBundleList.stream().map(x->x.getPerQuerySARE()).collect(Collectors.toList());
        double[] mean_sare = new double[perQuerySAREValuesList.get(0).length];
        for (double[] perQuerySAREValues: perQuerySAREValuesList) {
            for (int j=0; j < perQuerySAREValues.length; j++) {
                mean_sare[j] += perQuerySAREValues[j];
            }
            mean_sare = Arrays.stream(mean_sare).map(x->x/perQuerySAREValues.length).toArray();
        }

        return new QPPMetricBundle(tau_mean, ndcg_mean, mean_sare);
    }

    static QPPMetricBundle evaluate(List<MsMarcoQuery> queries,
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
        return new QPPMetricBundle(evaluatedMetricValues, qppEstimates);
    }

    public static double[] runExperiment(
            EvalMetricTieBreaker evalMetricTieBreaker,
            Metric targetMetric, int mode)
    throws Exception {

        List<MsMarcoQuery> queries;
        final String resFile =
                Constants.BM25_Top100_DL1920
                //Constants.ColBERT_Top100_DL1920
                ;
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

        Evaluator evaluator = new Evaluator(Constants.QRELS_DL1920, resFile, NUM_DOCS); // Metrics for top-100 (P@10 is still at 10)
        List<QPPMethod> evaluatedQPPModels = new ArrayList<>();

        double[] evaluatedMetricValues = new double[queries.size()];
        int i=0;
        for (MsMarcoQuery query: queries) {
            evaluatedMetricValues[i++] = evaluator.compute(query.getId(), targetMetric);
        }
        //System.out.println(frequencyMap(Arrays.stream(evaluatedMetricValues).boxed()));

        for (QPPMethod qppMethod: qppMethods) {
            QPPMetricBundle qppMetrics = evaluate(
                    queries, qppMethod,
                    evaluator, evaluatedMetricValues,
                    evalMetricTieBreaker
            );
//            System.out.println(String.format("%s on %s: tau = %.4f, sare = %.4f",
//                    qppMethod.name(),
//                    targetMetric.name(),
//                    qppMetrics.tau(), qppMetrics.sare()));

            qppMethod.setMeasure(qppMetrics);
            evaluatedQPPModels.add(qppMethod);
        }

//        List<QPPMethod> modelsRankedByPerfMeasure =
//            evaluatedQPPModels.stream().sorted(
//                (o1, o2)->
//                Double.compare(o1.getMeasure().tau(), o2.getMeasure().tau())
//            )
//            .collect(Collectors.toList());

//        for (QPPMethod qppModel: modelsRankedByPerfMeasure) {
//            System.out.println(String.format("%s: %.4f", qppModel.name(), qppModel.getMeasure().tau()));
//        }

        return mode==TAU?
            evaluatedQPPModels.stream()
                .mapToDouble(x -> x.getMeasure().tau())
                .toArray():
            evaluatedQPPModels.stream()
                    .mapToDouble(x -> x.getMeasure().ndcg())
                    .toArray()
        ;
    }

    public static double findCorrelation(EvalMetricTieBreaker evalMetricTieBreaker, Metric a, Metric b, int mode) throws Exception {
        double[] runA = runExperiment(evalMetricTieBreaker, a, mode);
        double[] runB = runExperiment(evalMetricTieBreaker, b, mode);

        return (new KendallsCorrelation()).correlation(runA, runB);
    }

    public static void findCorrelationsBetweenMetrics(EvalMetricTieBreaker tieBreaker) {
        try {
            //System.out.println("Evaluation w/o breaking ties");
            System.out.println(
                    String.format("Kendall's between the predictor rankings ordered by tau = %.4f",
                            findCorrelation(tieBreaker, Metric.P_10, Metric.AP, TAU))
            );

            //System.out.println("Evaluation w/ breaking ties");
            System.out.println(
                    String.format("Kendall's between the predictor rankings ordered by ndcg = %.4f",
                            findCorrelation(tieBreaker, Metric.P_10, Metric.AP, NDCG))
            );
        }
        catch (Exception e) { e.printStackTrace();}
    }

    public static void findCorrelationBetweenTieResolvers(int mode) {
        try {
            String modename = mode==TAU?"tau":"ndcg";
            double[] sortedQPPMeasures_notiebreaks = runExperiment(new NoTieBreaker(), Metric.P_10, mode);
            double[] sortedQPPMeasures_withtiebreaks = runExperiment(new NoisePerturbationTieBreaker(NUM_RANKINGS, DELTA), Metric.P_10, mode);

            System.out.println(
                    String.format("Kendall's between the predictors ranked by %s = %.4f", modename,
                            (new KendallsCorrelation())
                                    .correlation(sortedQPPMeasures_notiebreaks, sortedQPPMeasures_withtiebreaks)
                    )
            );
        }
        catch (Exception e) { e.printStackTrace();}
    }

    public static void main(String[] args) {
        findCorrelationsBetweenMetrics(new NoTieBreaker());

        findCorrelationBetweenTieResolvers(TAU);
        findCorrelationBetweenTieResolvers(NDCG);

    }
}
