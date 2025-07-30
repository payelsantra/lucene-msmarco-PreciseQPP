package qpp;
import retrieval.Constants;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

public class QueryVecLoader {
    public static Map<Long, double[]> loadEmbeddings(String path) throws IOException {
        Map<Long, double[]> embeddings = new HashMap<>();

        try (FileInputStream fis = new FileInputStream(path);
             FileChannel channel = fis.getChannel()) {

            // Each record is 8 (ID) + 768*8 (vector) = 6152 bytes
            ByteBuffer buffer = ByteBuffer.allocate(6152);
            buffer.order(ByteOrder.LITTLE_ENDIAN);  // PyTorch writes in little-endian

            while (channel.read(buffer) == 6152) {
                buffer.flip();

                // Read ID (int64)
                long id = buffer.getLong();

                // Read 768 float64 values
                double[] vec = new double[768];
                for (int i = 0; i < 768; i++) {
                    vec[i] = buffer.getDouble();
                }

                embeddings.put(id, vec);
                buffer.clear();
            }
        }

        return embeddings;
    }

    // Example usage
    public static void main(String[] args) throws IOException {
        Map<Long, double[]> embeddings = loadEmbeddings(Constants.DL20_CONTRIEVER_VECS);

        // Print one example
        embeddings.entrySet().stream().forEach(entry -> {
            System.out.println(entry.getKey() + Arrays.toString(Arrays.copyOf(entry.getValue(), 10)));
        });


    }
}
