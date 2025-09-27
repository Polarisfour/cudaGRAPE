#include "task.cuh"
#include "point.cuh"

// Parameterized constructor
__host__ __device__ Task::Task(int id, Point coordinates, int reward, int des) {
    this->id = id;
    this->coordinates = coordinates;
    this->reward = reward;
    this->des = des;
}

__host__ __device__ Point Task::getCoordinates() {
    return this->coordinates;
}

int Task::getId() {
    return this->id;
}

int Task::getReward() {
    return this->reward;
}

int Task::getDes() {
    return this->des;
}
