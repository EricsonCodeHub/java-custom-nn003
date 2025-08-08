import java.util.Random;

public class NeuralNetwork
{
    double[][][] nn; // Stores the NN weights
    double[][] bias; // Stores NN biases 
    double[][] xPerLayer; // Stores layers outputs for back propagation 
    double[][] xActPerLayer;  // stores activated outputs for back propagation
    private int activationType; // 0=ReLU, 1=tanh, 2=sigmoid, etc.


    /*
        Constructor 
    */
    public NeuralNetwork(int xInputs, int layers, int nodePerLayer, int activationType)
    {
        this.activationType = activationType; // store which activation to use

        nn = new double[layers][xInputs][xInputs];
        bias = new double[layers][xInputs];
        xPerLayer = new double[layers][xInputs];
        xActPerLayer = new double[layers][xInputs];

        Random rand = new Random();

        for (int k = 0; k < nn.length; k++)
        {
            for (int j = 0; j < nn[0].length; j++)
            {
                bias[k][j] = rand.nextDouble() * 0.2 - 0.1;
                for (int i = 0; i < nn[0][0].length; i++)
                {
                    nn[k][j][i] = rand.nextDouble() * 0.2 - 0.1;
                }
            }
        }
        System.out.println("nn created with activation type " + activationType);
    }

    /*
        Activation functions
    */ 
    private double activate(double x)
    {
        switch (activationType)
        {
            case 0: // ReLU
                return Math.max(0, x);
            case 1: // tanh
                return Math.tanh(x);
            case 2: // Sigmoid
                return 1.0 / (1.0 + Math.exp(-x));
            default:
                return x; // linear
        }
    }

    /*
        Derivatives for backprop
    */ 
    private double activateDerivative(double x)
    {
        switch (activationType)
        {
            case 0: // ReLU
                return (x > 0) ? 1 : 0;
            case 1: // tanh
                double t = Math.tanh(x);
                return 1 - t * t;
            case 2: // Sigmoid
                double s = 1.0 / (1.0 + Math.exp(-x));
                return s * (1 - s);
            default:
                return 1; // linear
        }
    }

    /*
        forwardPass
    */
    public double forwardPass(double[] x)
    {
        for(int k = 0; k < nn.length; k++)
        {
            for(int j = 0; j < nn[0].length; j++)
            {
                double curentNode = 0;

                for(int i = 0; i<nn[0][0].length; i++)
                {
                    curentNode += x[i] * nn[k][j][i];
                }

                xPerLayer[k][j] = curentNode + bias[k][j];
                xActPerLayer[k][j] = activate(xPerLayer[k][j]);
            }

            x = xActPerLayer[k];
        }

        double y = 0;
        for(int i = 0; i < nn[0].length; i++)
        {
            y += xActPerLayer[xActPerLayer.length - 1][i];
        }

        return y;
    }

    /*
        train
    */
    public double train(double[] x, double y, double lr)
    {
        double p = forwardPass(x);
        double error = p - y;
        double loss = error * error;

        double[][] delta = new double[nn.length][nn[0].length];

        // Output layer delta
        for (int j = 0; j < nn[0].length; j++)
        {
            delta[nn.length - 1][j] = 2 * error * activateDerivative(xPerLayer[nn.length - 1][j]);
        }

        // Hidden layers delta
        for (int k = nn.length - 2; k >= 0; k--)
        {
            for (int i = 0; i < nn[0].length; i++)
            {
                double sum = 0;
                for (int j = 0; j < nn[0].length; j++)
                {
                    sum += delta[k + 1][j] * nn[k + 1][j][i];
                }
                delta[k][i] = sum * activateDerivative(xPerLayer[k][i]);
            }
        }

        // Update weights
        for (int k = 0; k < nn.length; k++)
        {
            double[] inputToLayer = (k == 0) ? x : xActPerLayer[k - 1];
            for (int j = 0; j < nn[0].length; j++)
            {
                bias[k][j] -= lr * delta[k][j];
                for (int i = 0; i < nn[0][0].length; i++)
                {
                    nn[k][j][i] -= lr * delta[k][j] * inputToLayer[i];
                }
            }
        }

        return loss;
    }

    /*
        trainBatch

        basically averages graident decent
        i didnt write 
    */
    public double trainBatch(double[][] xBatch, double[] yBatch, double lr)
    {
        int batchSize = xBatch.length;
        int layersCount = nn.length;
        int nodesCount = nn[0].length;
        int inputSize = nn[0][0].length;

        // Arrays to accumulate gradients over the batch
        double[][] deltaSum = new double[layersCount][nodesCount]; // (not strictly needed separately)
        double[][][] weightGradSum = new double[layersCount][nodesCount][inputSize];
        double[][] biasGradSum = new double[layersCount][nodesCount];

        double totalLoss = 0;

        // To store forward pass intermediates per example
        double[][][] batch_xPerLayer = new double[batchSize][layersCount][nodesCount];
        double[][][] batch_xActPerLayer = new double[batchSize][layersCount][nodesCount];

        // For each example in the batch
        for (int b = 0; b < batchSize; b++) {
            double[] x = xBatch[b];
            double y = yBatch[b];

            // Forward pass modified to fill per-example arrays
            double[] input = x;
            for (int k = 0; k < layersCount; k++) {
                for (int j = 0; j < nodesCount; j++) {
                    double curentNode = 0;
                    for (int i = 0; i < inputSize; i++) {
                        curentNode += input[i] * nn[k][j][i];
                    }
                    batch_xPerLayer[b][k][j] = curentNode + bias[k][j];
                    batch_xActPerLayer[b][k][j] = activate(batch_xPerLayer[b][k][j]);
                }
                input = batch_xActPerLayer[b][k];
            }

            // Calculate predicted output as sum of last layer activations (same as forwardPass)
            double p = 0;
            for (int i = 0; i < nodesCount; i++) {
                p += batch_xActPerLayer[b][layersCount - 1][i];
            }

            double error = p - y;
            totalLoss += error * error;

            // Calculate deltas
            double[][] delta = new double[layersCount][nodesCount];

            // Output layer delta
            for (int j = 0; j < nodesCount; j++) {
                delta[layersCount - 1][j] = 2 * error * activateDerivative(batch_xPerLayer[b][layersCount - 1][j]);
            }

            // Hidden layers delta
            for (int k = layersCount - 2; k >= 0; k--) {
                for (int i = 0; i < nodesCount; i++) {
                    double sum = 0;
                    for (int j = 0; j < nodesCount; j++) {
                        sum += delta[k + 1][j] * nn[k + 1][j][i];
                    }
                    delta[k][i] = sum * activateDerivative(batch_xPerLayer[b][k][i]);
                }
            }

            // Accumulate gradients
            for (int k = 0; k < layersCount; k++) {
                double[] inputToLayer = (k == 0) ? x : batch_xActPerLayer[b][k - 1];
                for (int j = 0; j < nodesCount; j++) {
                    biasGradSum[k][j] += delta[k][j];
                    for (int i = 0; i < inputSize; i++) {
                        weightGradSum[k][j][i] += delta[k][j] * inputToLayer[i];
                    }
                }
            }
        }

        // Update weights and biases using averaged gradients
        for (int k = 0; k < layersCount; k++) {
            for (int j = 0; j < nodesCount; j++) {
                bias[k][j] -= lr * (biasGradSum[k][j] / batchSize);
                for (int i = 0; i < inputSize; i++) {
                    nn[k][j][i] -= lr * (weightGradSum[k][j][i] / batchSize);
                }
            }
        }

        return (totalLoss / batchSize);
    }



    public double[][] normalizeStandardDev2D(double[][] data)
    {
        int numSamples = data.length;
        if (numSamples == 0) return new double[0][0];
        int numFeatures = data[0].length;

        double[] means = new double[numFeatures];
        double[] stdDevs = new double[numFeatures];

        // Calculate means for each feature
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < numSamples; i++) {
                sum += data[i][j];
            }
            means[j] = sum / numSamples;
        }

        // Calculate standard deviations for each feature
        for (int j = 0; j < numFeatures; j++) {
            double varianceSum = 0;
            for (int i = 0; i < numSamples; i++) {
                double diff = data[i][j] - means[j];
                varianceSum += diff * diff;
            }
            stdDevs[j] = Math.sqrt(varianceSum / numSamples);
        }

        // Normalize the data
        double[][] normalized = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (stdDevs[j] == 0) {
                    normalized[i][j] = 0; // Avoid divide by zero
                } else {
                    normalized[i][j] = (data[i][j] - means[j]) / stdDevs[j];
                }
            }
        }

        return normalized;
    }
}
