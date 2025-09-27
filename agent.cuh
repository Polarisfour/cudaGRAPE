#ifndef AGENT_CUH
#define AGENT_CUH

#include "point.cuh"

// CUDA-friendly Agent class
class Agent {
public:
    // Constructors
    __host__ __device__ Agent(Point coordinates, int id, int task_id);
    __host__ __device__ Agent();

    // Getters and setters
    __host__ __device__ Point getCoordinates();
    __host__ __device__ int getId();
    __host__ __device__ int getTaskId();
    __host__ __device__ void setTaskId(int id);

private:
    int id;          // agent ID
    Point coordinates; // location of agent
    int task_id;     // current task assigned
};

#endif
