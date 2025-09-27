#ifndef TASK_CUH
#define TASK_CUH

#include "point.cuh"

// CUDA-friendly Task class
class Task {
public:
    __host__ __device__ Task();
    __host__ __device__ Task(int id, Point coordinates, int reward, int des);

    __host__ __device__ Point getCoordinates();
    __host__ __device__ int getReward();
    __host__ __device__ int getId();
    __host__ __device__ int getDes();

private:
    int id;          // task id
    Point coordinates; // location of task
    int reward;      // reward for completing task
    int des;         // desired coalition size
};

#endif
