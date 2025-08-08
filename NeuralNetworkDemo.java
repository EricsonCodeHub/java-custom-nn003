import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNetworkDemo {
    public static void main(String[] args) {
        String filePath = "data.csv";

        ArrayList<double[]> rows = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.trim().split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i].trim());
                }
                rows.add(row);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }

        int numRows = rows.size();
        int numCols = rows.get(0).length;

        double[][] data = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            data[i] = rows.get(i);
        }

        data = normalizeStandardDev2DMain(data);

        // Extract target/output (first column)
        double[] y = extractY(data);

        // Extract input features (all columns except first)
        double[][] x = extractX(data);

        System.out.println("y (target/output):");
        for (double v : y) {
            System.out.println(v);
        }

        System.out.println("\nx (input features):");
        for (double[] row : x) {
            for (double v : row) {
                System.out.print(v + " ");
            }
            System.out.println();
        }

        System.out.println("Data loaded and normalized.");

        // Neural network input size = number of features
        NeuralNetwork nn = new NeuralNetwork(x[0].length, 3, 10, 1); // 3 layers, 10 nodes each, tanh activation (1)

        int epochs = 5000;
        double lr = 0.0001;
        double loss = 1;

        for (int i = 0; i < epochs; i++) {
            loss = nn.trainBatch(x, y, lr);
            if (i % 100 == 0 || i == epochs - 1) {
                System.out.println("Epoch " + i + " average loss: " + loss);
            }
            if (loss < 0.001) {
                System.out.println("Loss below threshold, stopping training.");
                break;
            }
        }

        // Predictions
        System.out.println("\nPredictions after training:");
        for (int i = 0; i < Math.min(10, x.length); i++) {
            double prediction = nn.forwardPass(x[i]);
            System.out.printf("Input %d, Predicted: %.5f, Actual: %.5f%n", i, prediction, y[i]);
        }
    }

    public static double[] extractY(double[][] data) {
        int rows = data.length;
        double[] y = new double[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = data[i][0];
        }
        return y;
    }

    public static double[][] extractX(double[][] data) {
        int rows = data.length;
        int cols = data[0].length - 1;
        double[][] x = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 1, x[i], 0, cols);
        }
        return x;
    }

    public static double[][] normalizeStandardDev2DMain(double[][] data) {
        int numSamples = data.length;
        if (numSamples == 0) return new double[0][0];
        int numFeatures = data[0].length;

        double[] means = new double[numFeatures];
        double[] stdDevs = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < numSamples; i++) {
                sum += data[i][j];
            }
            means[j] = sum / numSamples;
        }

        for (int j = 0; j < numFeatures; j++) {
            double varianceSum = 0;
            for (int i = 0; i < numSamples; i++) {
                double diff = data[i][j] - means[j];
                varianceSum += diff * diff;
            }
            stdDevs[j] = Math.sqrt(varianceSum / numSamples);
        }

        double[][] normalized = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (stdDevs[j] == 0) {
                    normalized[i][j] = 0;
                } else {
                    normalized[i][j] = (data[i][j] - means[j]) / stdDevs[j];
                }
            }
        }

        return normalized;
    }
}
