#include "agent.cuh"

// Constructor with parameters
__host__ __device__ Agent::Agent(Point coordinates, int id, int task_id) {
    this->coordinates = coordinates;
    this->id = id;
    this->task_id = task_id;
}

// Default constructor at origin with id=0
__host__ __device__ Agent::Agent() {
    Point p;
    p.x = 0;
    p.y = 0;
    this->coordinates = p;
    this->id = 0;
    this->task_id = 0;
}

__host__ __device__ Point Agent::getCoordinates() {
    return this->coordinates;
}

__host__ __device__ int Agent::getId() {
    return this->id;
}

__host__ __device__ int Agent::getTaskId() {
    return this->task_id;
}

__host__ __device__ void Agent::setTaskId(int id) {
    this->task_id = id;
}
