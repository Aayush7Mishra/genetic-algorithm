#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include "NeuralNetwork.h"

struct Individual {
    std::vector<double> weights;
    double fitness;
};

class GeneticAlgorithm {
public:
    int populationSize, generations;
    double mutationRate;
    NeuralNetwork& nn;
    std::vector<Individual> population;

    GeneticAlgorithm(int populationSize, int generations, double mutationRate, NeuralNetwork& nn)
        : populationSize(populationSize), generations(generations), mutationRate(mutationRate), nn(nn) {
        initializePopulation();
    }

    void initializePopulation() {
        for (int i = 0; i < populationSize; ++i) {
            Individual ind;
            ind.weights = nn.getWeights();
            for (double& weight : ind.weights)
                weight = ((double)rand() / (RAND_MAX)) * 2 - 1;
            ind.fitness = 0.0;
            population.push_back(ind);
        }
    }

    double fitnessFunction(const std::vector<double>& weights) {
        nn.setWeights(weights);
        // Example: simple fitness function for XOR problem
        std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        std::vector<double> outputs = { 0, 1, 1, 0 };
        double fitness = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output = nn.feedForward(inputs[i]);
            fitness += pow(output[0] - outputs[i], 2);
        }
        return -fitness; // Minimize error
    }

    void evaluateFitness() {
        for (Individual& ind : population)
            ind.fitness = fitnessFunction(ind.weights);
    }

    Individual selectParent() {
        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
            });
        return population[0]; // Elitism: always select the best
    }

    void mutate(Individual& ind) {
        for (double& weight : ind.weights) {
            if (((double)rand() / (RAND_MAX)) < mutationRate)
                weight += ((double)rand() / (RAND_MAX)) * 2 - 1;
        }
    }

    void evolve() {
        for (int gen = 0; gen < generations; ++gen) {
            evaluateFitness();
            std::vector<Individual> newPopulation;
            while (newPopulation.size() < populationSize) {
                Individual parent = selectParent();
                Individual offspring = parent;
                mutate(offspring);
                newPopulation.push_back(offspring);
            }
            population = newPopulation;
            std::cout << "Generation " << gen << " Best Fitness: " << population[0].fitness << std::endl;
        }
    }
};

#endif 

