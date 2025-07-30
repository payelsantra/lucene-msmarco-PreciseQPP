package qpp;

import opennlp.tools.parser.Cons;
import retrieval.Constants;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

public class ChunkedMMapEmbeddingReader {
    private static final int VECTOR_SIZE = 768;
    private static final int RECORD_SIZE = 8 + VECTOR_SIZE * 8; // 6152 bytes

    private final List<MappedByteBuffer> chunks;
    private final int recordsPerChunk;
    private final long totalRecords;

    public ChunkedMMapEmbeddingReader(String filePath, int recordsPerChunk) throws IOException {
        this.recordsPerChunk = recordsPerChunk;
        this.chunks = new ArrayList<>();

        File file = new File(filePath);
        long fileSize = file.length();
        this.totalRecords = fileSize / RECORD_SIZE;

        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            FileChannel channel = raf.getChannel();

            long numChunks = (totalRecords + recordsPerChunk - 1) / recordsPerChunk;
            for (int i = 0; i < numChunks; i++) {
                long chunkStartOffset = (long) i * recordsPerChunk * RECORD_SIZE;
                long chunkSize = Math.min(recordsPerChunk * (long) RECORD_SIZE, fileSize - chunkStartOffset);

                MappedByteBuffer chunkBuffer = channel.map(FileChannel.MapMode.READ_ONLY, chunkStartOffset, chunkSize);
                chunkBuffer.order(ByteOrder.LITTLE_ENDIAN);
                chunks.add(chunkBuffer);
            }
        }
    }

    public double[] getVector(long docId) {
        if (docId < 0 || docId >= totalRecords) {
            throw new IndexOutOfBoundsException("docId " + docId + " is out of bounds.");
        }

        int chunkIndex = (int) (docId / recordsPerChunk);
        int recordInChunk = (int) (docId % recordsPerChunk);
        int offsetInChunk = recordInChunk * RECORD_SIZE;

        ByteBuffer chunkBuffer = chunks.get(chunkIndex).duplicate();
        chunkBuffer.order(ByteOrder.LITTLE_ENDIAN);
        chunkBuffer.position(offsetInChunk);

        long storedId = chunkBuffer.getLong();
        if (storedId != docId) {
            throw new IllegalStateException("Expected docId " + docId + ", found " + storedId);
        }

        double[] vec = new double[VECTOR_SIZE];
        for (int i = 0; i < VECTOR_SIZE; i++) {
            vec[i] = chunkBuffer.getDouble();
        }
        return vec;
    }

    public long getTotalRecords() {
        return totalRecords;
    }

    // Example usage
    public static void main(String[] args) throws IOException {
        ChunkedMMapEmbeddingReader reader =
                new ChunkedMMapEmbeddingReader(Constants.COLL_DENSEVEC_FILE_CONTRIEVER, Constants.RECORCDS_PER_CHUNK);

        long docId = 1L;
        double[] vec = reader.getVector(docId);
        System.out.println("Vector[0:5] for docId " + docId + ": " + Arrays.toString(Arrays.copyOf(vec, 5)));
    }
}
