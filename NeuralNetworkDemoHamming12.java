import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetworkDemoHamming12 {

    public static void main(String[] args) {
        String filePath = "hamming12_data.csv";

        ArrayList<double[]> rows = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(filePath))) {
            // Skip header
            if (scanner.hasNextLine()) scanner.nextLine();

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.trim().split(",");

                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                rows.add(row);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return; // Exit if file not found
        }

        int numRows = rows.size();
        int numCols = rows.get(0).length;

        double[][] data = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            data[i] = rows.get(i);
        }

        // Extract label column (last column)
        double[] y = new double[numRows];
        // Extract input features (all except last)
        double[][] x = new double[numRows][numCols - 1];

        for (int i = 0; i < numRows; i++) {
            y[i] = data[i][numCols - 1];
            System.arraycopy(data[i], 0, x[i], 0, numCols - 1);
        }

        System.out.println("Data loaded. Total samples: " + numRows);

        // Initialize Neural Network
        // Input size: 12 bits, 2 hidden layers with 16 nodes each, output layer size 1
        int inputSize = 12;
        int layers = 12;
        int nodesPerLayer = 12;
        int activationType = 2;  // Sigmoid

        NeuralNetwork nn = new NeuralNetwork(inputSize, layers, nodesPerLayer, activationType);

        double learningRate = 0.1;
        int epochs = 10000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = nn.trainBatch(x, y, learningRate);
            if (epoch % 100 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, loss: %.6f%n", epoch, loss);
            }
            if (loss < 0.06) {
                System.out.println("Loss below threshold, stopping training.");
                break;
            }
        }

        // Test predictions on first 20 samples
        System.out.println("\nSample predictions after training:");
        for (int i = 0; i < Math.min(20, x.length); i++) {
            double pred = nn.forwardPass(x[i]);
            System.out.printf("Sample %d, Predicted: %.4f, Actual: %.0f%n", i, pred, y[i]);
        }
    }
}
