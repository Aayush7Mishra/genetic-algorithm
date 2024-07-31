#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include "NeuralNetwork.h"

struct position{
    int x;
    int y;
    position(float x=0, float y=0){
        this->x = x;
        this->y = y;
    }
};

int random_int(int start=0, int end=100){
    
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // // Define the uniform integer distribution for numbers between start and end
    // std::uniform_int_distribution<> dis(start, end);
    // double random_number = dis(gen);

    double random_number = ((double)rand()/ RAND_MAX) * (end - start) + start;

    return random_number;
}


int id = 0;

const int MAP_X = 100;
const int MAP_Y = 100;

position food_array[5] = {
    position(0, 5),
    position(20, 20),
    position(33, 33),
    position(66, 78),
    position(22, 78),
    
};

void randomizeFoodPositions() {
    for (int i = 0; i < 5; i++) {
        food_array[i].x = random_int(0,80);
        food_array[i].y = random_int(0,80);
    }
}
class Individual{
    public:

        std::vector<double> weights;
        std::vector<int> active_connections;

        float fitness=0;
        unsigned short id;
        int dead = 0;
        
        position pos;
        position food;
        int food_dir;
        int new_species;
        int species_id = 1;


        std::vector<position> movement_history;
        std::vector<position> food_history;
        std::vector<float> fitness_history;

        int direction = 0; // 0 = up-left, 1 = up, 2 = up-right, 3 = left, 4 = right, 5 = down-left, 6 = down, 7 = down-right

        Individual(){
            id++;
            pos = position(0,0);
            food = position(20,20);
            food_dir = 0;
            new_species = 0;
            
        }

        void clean(){
            fitness = 0.0;
            pos = position(random_int(),random_int());
            food = position(random_int(0,80),random_int(0,80));
            food_dir = 0;//0->top-left,1->top,2->top-right,3->left,4->right,5->bottom-left,6->bottom,7->bottom-right
            direction = 0;
            movement_history.clear();
            fitness_history.clear();
            food_history.clear();
        }


        void move (std::vector<double>&  outputs){
            movement_history.push_back(pos);

            food_history.push_back(food);
            direction = 0;
            if (outputs[0] < -.5){;
                direction++;
                if (pos.x > 0)
                    pos.x -= 1;
            }
            else if (outputs[0] > .5){
                direction--;
                if (pos.x < MAP_X - 1)
                    pos.x += 1;
            }

            if (outputs[1] < -0.5){
                direction+=5;
                if (pos.y > 0)
                    pos.y -= 1;
            }
            else if (outputs[1] > 0.5){
                if (pos.y < MAP_Y - 1)
                    pos.y += 1;
            }
            
        }

        // void move(int next_direction){
        //     direction = next_direction;
        //     if (0 == dead){
        //         if (next_direction == 0){
        //             if (pos.x > 0 && pos.y > 0) {
        //                 pos.x -= 1;
        //                 pos.y -= 1;
        //             }
        //         }
        //         else if (next_direction == 1){
        //             if (pos.y > 0) {
        //                 pos.y -= 1;
        //             }
        //         }
        //         else if (next_direction == 2){
        //             if (pos.x < MAP_X - 1 && pos.y > 0) {
        //                 pos.x += 1;
        //                 pos.y -= 1;
        //             }
        //         }
        //         else if (next_direction == 3){
        //             if (pos.x > 0) {
        //                 pos.x -= 1;
        //             }
        //         }
        //         else if (next_direction == 4){
        //             if (pos.x < MAP_X - 1) {
        //                 pos.x += 1;
        //             }
        //         }
        //         else if (next_direction == 5){
        //             if (pos.x > 0 && pos.y < MAP_Y - 1) {
        //                 pos.x -= 1;
        //                 pos.y += 1;
        //             }
        //         }
        //         else if (next_direction == 6){
        //             if (pos.y < MAP_Y - 1) {
        //                 pos.y += 1;
        //             }
        //         }
        //         else if (next_direction == 7){
        //             if (pos.x < MAP_X - 1 && pos.y < MAP_Y - 1) {
        //                 pos.x += 1;
        //                 pos.y += 1;
        //             }
        //         }
        //         else if (next_direction == 8){
        //             // No movement
        //         }
        //     }
        // }

        
};
#endif