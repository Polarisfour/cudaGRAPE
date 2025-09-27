/*
CUDA Implementation of the GRAPE Coalition Formation Algorithm
Jacob Tanchak – April 25, 2025

This program uses CUDA to implement the GRAPE coalition formation algorithm.
Each agent is represented by a CUDA thread and iteratively decides which task
(coalition) to join based on a reward function.
*/

#include "task.cuh"
#include "point.cuh"
#include "agent.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <iostream>
#include <math.h>

// Adjust these constants for testing – beware of GPU memory limits at high values
const int a = 10;  // number of agents
const int t = 4;   // number of tasks

/*
Reward function using peaked reward.
Returns utility of either current coalition, or a prospective coalition.
*/
__device__ float reward(Agent agent, Task task, int count) {
    float reward_max = task.getReward();
    Point task_loc = task.getCoordinates();
    Point agent_loc = agent.getCoordinates();
    int des = task.getDes();
    int size = count;
    float dist = sqrt(pow(agent_loc.x - task_loc.x, 2) + pow(agent_loc.y - task_loc.y, 2));
    if (agent.getTaskId() != task.getId()) {
        size += 1;
    }
    float temp = reward_max * size / des;
    float temp2 = exp((-1.0 * size / des) + 1.0);
    int reward = temp * temp2;
    return reward - dist;
}

/*
Main decision kernel function. Each thread represents one agent and collectively
they adjust coalition membership until stable.
*/
__global__ void decision(Task* tasks, Agent* agents) {
    // Shared array for how many agents are satisfied with their coalition
    __shared__ int satcount[a];
    // Shared array for communication between agents
    __shared__ int share_array[a * (a + t + 2)];
    __syncthreads();

    // Local partition of agents into tasks, initialized to -1
    int partition[t][a];
    for (int i = 0; i < t; i++) {
        for (int j = 0; j < a; j++) {
            partition[i][j] = -1;
        }
    }

    // Fill the null coalition (task 0) with all agent IDs initially
    for (int i = 0; i < a; i++) {
        partition[0][i] = agents[i].getId();
    }

    // Track the size of all coalitions
    int coalition_sizes[t];
    for (int i = 0; i < t; i++) {
        coalition_sizes[i] = 0;
    }
    coalition_sizes[0] = a;

    // Initial agent values for satisfaction and iteration counters
    bool satisfied = false;
    int r_k = 0;  // iteration count
    int s_k = 0;  // random tie-breaker
    int sum;

    // Each thread is one agent
    int t_idx = threadIdx.x;
    Agent agent = agents[t_idx];

    // Offsets into shared array for communication between threads
    int offset = agent.getId() * (t + a + 2);
    int read_offset;
    if (agent.getId() != a - 1) {
        read_offset = (agent.getId() + 1) * (a + 2 + t);
    } else {
        read_offset = 0;
    }

    // Variables for comparing rewards of each coalition
    float temp_reward;
    float max_reward = 0;
    int max_coalition = 0;
    float current_reward;
    int current_coalition;
    bool temp_0 = true;
    bool temp_1 = true;

    // Main loop – continues until all agents are satisfied
    while (true) {
        if (!satisfied) {
            // Calculate reward for each coalition and pick the best
            max_reward = 0;
            for (int i = 0; i < t; i++) {
                temp_reward = reward(agent, tasks[i], coalition_sizes[i]);
                if (agent.getTaskId() != tasks[i].getId()) {
                    if (temp_reward > max_reward) {
                        max_reward = temp_reward;
                        max_coalition = i;
                    }
                } else {
                    current_reward = temp_reward;
                    current_coalition = i;
                }
            }
            // Switch to a better coalition if found
            if ((max_reward > current_reward) && (current_coalition != max_coalition)) {
                agent.setTaskId(max_coalition);
                coalition_sizes[current_coalition]--;
                coalition_sizes[max_coalition]++;
                // Update partition array
                for (int i = 0; i < a; i++) {
                    if (partition[current_coalition][i] == agent.getId() && temp_0) {
                        partition[current_coalition][i] = -1;
                        temp_0 = false;
                    }
                    if (partition[max_coalition][i] == -1 & temp_1) {
                        partition[max_coalition][i] = agent.getId();
                        temp_1 = false;
                    }
                    if ((!temp_0) && (!temp_1)) {
                        temp_0 = true;
                        temp_1 = true;
                        break;
                    }
                }
                // Increase iterations and generate random tie-breaker
                r_k += 1;
                curandState state;
                curand_init(static_cast<int>(clock()), t_idx, 0, &state);
                s_k = curand(&state) % 100;
            }
            satisfied = true;
            satcount[t_idx] = 1;
        }

        // Compress partition to send via shared memory
        int partition_index = 0;
        int compressed_partition[a];
        int delimiters[t];
        for (int i = 0; i < t; i++) {
            delimiters[i] = 0;
        }
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < a; j++) {
                if (partition[i][j] != -1) {
                    compressed_partition[partition_index] = partition[i][j];
                    partition_index++;
                }
            }
        }
        delimiters[0] = coalition_sizes[0];
        for (int i = 1; i < t; i++) {
            delimiters[i] = delimiters[i - 1] + coalition_sizes[i];
        }

        __syncthreads();
        // Save local state to shared array
        share_array[offset] = r_k;
        share_array[offset + 1] = s_k;
        for (int i = 0; i < t; i++) {
            share_array[offset + 2 + i] = delimiters[i];
        }
        for (int i = 0; i < a; i++) {
            share_array[offset + 2 + i + t] = compressed_partition[i];
        }

        // Synchronize then read from next agent’s slot
        __syncthreads();
        int shared_r_k = share_array[read_offset];
        float shared_s_k = share_array[read_offset + 1];
        int start;

        // Choose the superior partition based on iteration and random tie-breaker
        if (shared_r_k > r_k || ((shared_r_k == r_k) && shared_s_k > s_k)) {
            // Fill partition with -1
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < a; j++) {
                    partition[i][j] = -1;
                }
            }
            // Copy info from shared array
            for (int i = read_offset + 2; i < t + read_offset + 2; i++) {
                delimiters[i - read_offset - 2] = share_array[i];
            }
            for (int i = read_offset + 2 + t; i < t + a + read_offset + 2; i++) {
                compressed_partition[i - read_offset - 2 - t] = share_array[i];
            }
            // Decompress the partition
            for (int i = 0; i < t; i++) {
                coalition_sizes[i] = 0;
                if (i == 0) {
                    start = 0;
                } else {
                    start = delimiters[i - 1];
                }
                coalition_sizes[i] = delimiters[i] - start;
                for (int j = start; j < delimiters[i]; j++) {
                    partition[i][j - start] = compressed_partition[j];
                    if (compressed_partition[j] == agent.getId()) {
                        agent.setTaskId(i);
                    }
                }
            }
            satisfied = false;
            satcount[t_idx] = 0;
            r_k = shared_r_k;
            s_k = shared_s_k;
        }

        // Check if all agents are satisfied
        sum = 0;
        __syncthreads();
        for (int i = 0; i < a; i++) {
            if (satcount[i] == 1) {
                sum += 1;
            }
        }
        if (sum == a) {
            break;
        }
    }

    // Print the resulting coalitions once finished
    if (t_idx == 0) {
        printf("Partition:\n");
        for (int i = 0; i < t; i++) {
            printf("Task: %d: {", i);
            for (int j = 0; j < a; j++) {
                if (partition[i][j] != -1) {
                    printf(" %d ", partition[i][j]);
                }
            }
            printf("}\n");
        }
        printf("Coalition sizes: ");
        for (int i = 0; i < t; i++) {
            printf(" %d ", coalition_sizes[i]);
        }
        printf("\nIterations: %d\n", r_k);
    }
}

/*
Main function:
Generates random agents and tasks, copies to GPU, launches kernel,
and measures time.
*/
int main() {
    // Host memory for agents and tasks
    Agent* agents = (Agent*)malloc(sizeof(Agent) * a);
    Task* tasks = (Task*)malloc(sizeof(Task) * t);

    // Device memory pointers
    Agent* c_agents;
    Task* c_tasks;

    // Allocate device memory
    cudaMalloc((void**)&c_agents, (sizeof(Agent) * a));
    cudaMalloc((void**)&c_tasks, (sizeof(Task) * t));

    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 10);
    std::uniform_int_distribution<> distr01(50, 250);
    std::uniform_int_distribution<> distr02(1, (a / t));

    // Fill agents with random positions
    for (int i = 0; i < a; i++) {
        Point p;
        p.x = distr(gen);
        p.y = distr(gen);
        agents[i] = Agent(p, i, 0);
    }

    // Fill tasks with random positions and rewards
    for (int j = 0; j < t; j++) {
        Point p;
        p.x = distr(gen);
        p.y = distr(gen);
        int reward = distr01(gen);
        if (j == 0) {
            reward = 0;
        }
        int des = distr02(gen);
        tasks[j] = Task(j, p, reward, des);
    }

    // Print tasks info
    printf("TASKS\n-------------------\n");
    for (int i = 0; i < t; i++) {
        printf("Task ID: %d\nTask Reward: %d\nDesired coalition size: %d\n----------------\n",
               tasks[i].getId(), tasks[i].getReward(), tasks[i].getDes());
    }

    // Copy to device memory
    cudaMemcpy(c_agents, agents, sizeof(Agent) * a, cudaMemcpyHostToDevice);
    cudaMemcpy(c_tasks, tasks, sizeof(Task) * t, cudaMemcpyHostToDevice);

    // Measure time and launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    decision<<<1, a>>>(c_tasks, c_agents);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time elapsed: " << duration.count() << " ms" << std::endl;
}
