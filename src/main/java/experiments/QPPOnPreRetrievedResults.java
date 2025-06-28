package experiments;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import qpp.*;
import qrels.AllRetrievedResults;
import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.QueryLoader;
import stochastic_qpp.*;
import utils.IndexUtils;

import java.io.*;
import java.util.*;

public class QPPOnPreRetrievedResults {
    // static final String BM25_MSMARCO_DEV_TOP100 = "runs/bm25_100_msmarcodev.res";

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Arguments expected: <query file> <TREC formatted res file>");
            args = new String[2];
            args[0] = Constants.QUERIES_DL20;
            args[1] = "../pyqppeval/data/runs/2020/BM25.2020.res";
        }

        String queryFile = args[0];
        String resFile = args[1];

        IndexReader reader = DirectoryReader.open(FSDirectory.open(new File(Constants.MSMARCO_INDEX).toPath()));
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());
        IndexUtils.init(searcher);

        Map<String, MsMarcoQuery> queryMap = QueryLoader.constructQueryMap(queryFile);
        AllRetrievedResults allRetrievedResults = new AllRetrievedResults(new File(resFile).getPath(), true);

        BufferedWriter bw = new BufferedWriter(new FileWriter(resFile + ".qpp"));

        final QPPMethod[] qppMethods = {
                new NQCSpecificity(searcher),
                new UEFSpecificity(new NQCSpecificity(searcher)),
                new RSDSpecificity(new NQCSpecificity(searcher)),
                new OddsRatioSpecificity(searcher, 0.4f),
                new WIGSpecificity(searcher),
                new NQCCalibratedSpecificity(searcher, 0.33f, 0.33f, 0.33f),
                new VariantSpecificity(
                        new NQCSpecificity(searcher),
                        searcher,
                        new KNNRelModel(Constants.QRELS_TRAIN, Constants.QUERY_FILE_TEST, false),
                        5, 0.2f
                ),
        };

        int count = 0;
        for (String qid: allRetrievedResults.queries()) {
            if (count++ % 1000 == 0)
                System.out.println(String.format("QPP completed for %d queries\r", count));

            StringBuilder sb = new StringBuilder();
            sb.append(qid).append("\t");
            for (QPPMethod qppMethod : qppMethods) {
                float qppEstimate = (float) qppMethod.computeSpecificity(
                        queryMap.get(qid),
                        allRetrievedResults.castToTopDocs(qid),
                        Constants.QPP_NUM_TOPK);
                sb.append(qppEstimate).append("\t");
            }
            sb.deleteCharAt(sb.length()-1);
            bw.write(sb.toString());
            bw.newLine();
        }

        bw.close();
    }
}
