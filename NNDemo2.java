public class NNDemo2 {

    public static void main(String[] args) {
        // XOR dataset
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] outputs = {0, 1, 1, 0};

        int inputSize = 2;
        int layers = 2;          // 2 layers: hidden + output
        int nodesPerLayer = 2;   // 2 nodes per layer
        int activationType = 2;  // Sigmoid activation

        NeuralNetwork nn = new NeuralNetwork(inputSize, layers, nodesPerLayer, activationType);

        double learningRate = 0.1;
        int epochs = 10000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < inputs.length; i++) {
                totalLoss += nn.train(inputs[i], outputs[i], learningRate);
            }

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Loss: " + totalLoss);
            }

            if(totalLoss < 0.001)
            {
                break;
            }
        }

        System.out.println("\nTesting after training:");
        for (int i = 0; i < inputs.length; i++) {
            double prediction = nn.forwardPass(inputs[i]);
            System.out.printf("Input: %s, Predicted: %.4f, Expected: %.1f%n",
                              java.util.Arrays.toString(inputs[i]), prediction, outputs[i]);
        }
    }
}
