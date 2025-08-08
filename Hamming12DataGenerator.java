import java.io.FileWriter;
import java.io.IOException;

public class Hamming12DataGenerator {

    public static void main(String[] args) {
        String filename = "hamming12_data.csv";

        try (FileWriter fw = new FileWriter(filename)) {
            // Header: 12 bits + label
            fw.write("b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,correct\n");

            // Generate all 12-bit codewords with single-bit errors allowed
            // For simplicity, just generate all 4096 possible 12-bit words
            // and label 1 if Hamming parity correct, else 0.

            for (int code = 0; code < 4096; code++) {
                int[] bits = intToBits(code, 12);

                int parity = calculateHammingParity(bits);

                // If parity == 0, code is correct, else code is corrupted
                int correct = (parity == 0) ? 1 : 0;

                // Write bits and label
                for (int i = 0; i < 12; i++) {
                    fw.write(bits[i] + ",");
                }
                fw.write(correct + "\n");
            }

            System.out.println("Data generated to " + filename);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Convert int to bits array (length bitsCount)
    public static int[] intToBits(int val, int bitsCount) {
        int[] bits = new int[bitsCount];
        for (int i = 0; i < bitsCount; i++) {
            bits[bitsCount - 1 - i] = (val >> i) & 1;
        }
        return bits;
    }

    // Calculate parity check for 12-bit Hamming (positions 1,2,4,8 parity bits)
    // Returns 0 if no error detected, else non-zero
    public static int calculateHammingParity(int[] bits) {
        // bits are zero-based index, positions 1-based
        // parity bits at positions 1,2,4,8 (1-based), i.e. indices 0,1,3,7 (0-based)
        // parity check covers these bits per Hamming(12,8)

        int p1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6] ^ bits[8] ^ bits[10];
        int p2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6] ^ bits[9] ^ bits[10];
        int p4 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6] ^ bits[11];
        int p8 = bits[7] ^ bits[8] ^ bits[9] ^ bits[10] ^ bits[11];

        // Syndrome is XOR of parity bits
        int syndrome = p1 + p2 + p4 + p8;

        return syndrome; // 0 means no error detected
    }
}
