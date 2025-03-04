package qpp;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import qrels.RetrievedResults;
import retrieval.MsMarcoQuery;

import java.util.Set;

public class WIGSpecificity extends BaseIDFSpecificity {

    public WIGSpecificity(IndexSearcher searcher) {
        super(searcher);
    }

    @Override
    public double computeSpecificity(MsMarcoQuery q, TopDocs topDocs, int k) {
        double[] rsvs = getRSVs(topDocs, k);
        double avgIDF = 0;
        int numQueryTerms = 1;
        try {
            Set<Term> qterms = q.getQueryTerms();
            numQueryTerms = qterms.size();
            avgIDF = 1/maxIDF(q.getQuery());
        }
        catch (Exception ex) { ex.printStackTrace(); }

        double wig = 0;
        for (double rsv: rsvs) {
            wig += (rsv - avgIDF);
        }
//        return wig/(double)(Math.sqrt(numQueryTerms) * rsvs.length);
        return wig/(double)(numQueryTerms * rsvs.length);
    }

    @Override
    public String name() {
        return "wig";
    }
}
