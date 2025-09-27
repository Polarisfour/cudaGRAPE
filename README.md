# CUDA GRAPE Coalition Formation Algorithm

This is a CUDA implementation of the GRAPE coalition formation algorithm.  
It assigns agents to tasks using a parallel decision-making process on the GPU.

---

## Overview

The code defines Agent and Task classes in CUDA and launches a `decision` kernel where each thread represents an agent.  
Agents iteratively adjust which task (coalition) they belong to based on rewards until a stable partition is reached.

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `nvcc` compiler
- curand library (for random numbers)

---

## File Structure

- `point.cuh` – simple 2D point struct
- `agent.cuh`/`agent.cu` – Agent class definition and implementation
- `task.cuh`/`task.cu` – Task class definition and implementation
- `main.cu` – contains CUDA kernel and `main()` function

---

## How to Build

Compile with `nvcc`, linking curand:

```bash
nvcc -std=c++14 main.cu agent.cu task.cu -o grape -lcurand
```

Make sure all `.cuh` and `.cu` files are in the same directory.

---

## How to Run

```bash
./grape
```

It will:

1. Randomly generate agents and tasks.
2. Copy them to device memory.
3. Launch the `decision` kernel.
4. Print out the resulting coalition partition and timing info.

---

## Adjusting Parameters

At the top of `main.cu`:

```cpp
const int a = 10; // number of agents
const int t = 4;  // number of tasks
```

Change these constants to test different scales.  
Beware of GPU memory limits for large values.

---

## Output

- Prints generated tasks and their rewards.
- Prints final partition of agents per task.
- Prints coalition sizes and total iterations.
- Prints elapsed time.

---

## License

Educational use only.
