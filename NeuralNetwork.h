#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Activation function
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
inline double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
public:
    int inputSize, hiddenSize, outputSize;
    std::vector<double> hiddenLayer, outputLayer;
    std::vector<std::vector<double>> weightsInputHidden, weightsHiddenOutput;

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        hiddenLayer.resize(hiddenSize);
        outputLayer.resize(outputSize);
        weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
        initializeWeights();
    }

    void initializeWeights() {
        srand(static_cast<unsigned>(time(0)));
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize; ++j)
                weightsInputHidden[i][j] = ((double)rand() / (RAND_MAX)) * 2 - 1;
        for (int i = 0; i < hiddenSize; ++i)
            for (int j = 0; j < outputSize; ++j)
                weightsHiddenOutput[i][j] = ((double)rand() / (RAND_MAX)) * 2 - 1;
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenLayer[i] = 0.0;
            for (int j = 0; j < inputSize; ++j)
                hiddenLayer[i] += inputs[j] * weightsInputHidden[j][i];
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }
        for (int i = 0; i < outputSize; ++i) {
            outputLayer[i] = 0.0;
            for (int j = 0; j < hiddenSize; ++j)
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
            outputLayer[i] = sigmoid(outputLayer[i]);
        }
        return outputLayer;
    }

    std::vector<double> getWeights() const {
        std::vector<double> weights;
        for (const auto& row : weightsInputHidden)
            weights.insert(weights.end(), row.begin(), row.end());
        for (const auto& row : weightsHiddenOutput)
            weights.insert(weights.end(), row.begin(), row.end());
        return weights;
    }

    void setWeights(const std::vector<double>& weights) {
        int index = 0;
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize; ++j)
                weightsInputHidden[i][j] = weights[index++];
        for (int i = 0; i < hiddenSize; ++i)
            for (int j = 0; j < outputSize; ++j)
                weightsHiddenOutput[i][j] = weights[index++];
    }
};

#endif 


