#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"

int main() {
    srand(static_cast<unsigned>(time(0)));
    NeuralNetwork nn(2, 4, 8);
    GeneticAlgorithm ga(50, 100, 0.05, nn);
    ga.evolve();
    return 0;
}


