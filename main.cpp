#include "NeuralNetwork.h"
#include "GeneticAlgorithm.h"
//#include "Renderer.h"
//#include <SDL2/SDL.h>
int main(int argc, char* args[]) {
    srand(static_cast<unsigned>(time(0)));
    FeedForwardNetwork nn(5, 4, 2);
    int populationSize = 500;
    GeneticAlgorithm ga(populationSize, 0.15, nn);
    
    int generations = 100;
    int step_per_gen = 400;
    /*SDL_Window* window=nullptr;
        SDL_Renderer* renderer=nullptr;
        SDL_Init(SDL_INIT_VIDEO);
        SDL_CreateWindowAndRenderer(100*8, 100*8+300, 0, &window, &renderer);
        SDL_RenderSetScale(renderer, 8, 8);
        SDL_SetRenderDrawColor(renderer,0,0,0,255);*/
    for (int gen = 0; gen < generations; ++gen) {
        randomizeFoodPositions();
        ga.evolve(gen, step_per_gen);
        /*Rend kk(ga.population);
        kk.draw2(renderer,gen);*/
        //for(int ind=0;ind<1;ind+=30){
            //kk.draw(renderer);
        // SDL_SetRenderDrawColor(renderer,0,0,0,255);
        // SDL_RenderClear(renderer);
        
        //     SDL_SetRenderDrawColor(renderer,255,255,255,255);
        //     for(int step=0;step<400 && gen>10;step++){
        //         SDL_RenderDrawPoint(renderer,ga.population[ind].movement_history[step].x,ga.population[ind].movement_history[step].y);
        //         if(step%20==0){SDL_RenderPresent(renderer);}
        //         std::cout<<"SP"<<step<<"turn"<<ind<<"x:"<<ga.population[ind].movement_history[step].x<<"y:"<<ga.population[ind].movement_history[step].y<<std::endl;
        //     }
    

        // }
        // SDL_Delay(500);
        
    
    }

    std::cout<<"best fitness: "<<ga.best.fitness<< " Gen: "<< ga.bestgen << " Id: "<<ga.best.species_id<<std::endl;
    std::cout<<"Weights shape: " << ga.best.weights.size() << std::endl;
    for (int i = 0; i < ga.best.active_connections.size(); ++i) {
        std::cout << ga.best.active_connections[i] << " ";
    }   
    for (int i = 0; i < ga.best.weights.size(); ++i) {
        std::cout << ga.best.weights[i] << " ";
    }
    
    std::cout << "Movement History of Best: " <<ga.bestgen<< std::endl;

    for (int i = 0; i < ga.best.movement_history.size(); i+=20) {
        std::cout<<i<<". " << ga.best.movement_history[i].x<< " "<<ga.best.movement_history[i].y<<" Fitness: "<<ga.best.fitness_history[i]<<std::endl;
    }
    return 0;
}


