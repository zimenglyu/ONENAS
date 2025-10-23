# ONE-NAS: Online NeuroEvolution-based Neural Architecture Search

ONE-NAS (Online NeuroEvolution-based Neural Architecture Search) is the first evolutionary algorithm capable of designing and training RNNs in real-time as data arrives in an online fashion. Unlike traditional time series forecasting methods that require offline pre-training, ONE-NAS continuously evolves both the structure and weights of Recurrent Neural Networks in response to streaming data. The algorithm utilizes island-based evolutionary strategies with repopulation techniques to maintain diversity and prevent catastrophic forgetting, while training new genomes on subsets of historical data to handle data drift effectively.

Implemented in C++ and built on the same foundation as EXAMM, ONE-NAS is designed for distributed computation and offers excellent scalability from personal laptops to high-performance computing clusters. The system employs a distributed architecture where worker processes handle RNN training while a main process manages population evolution and orchestrates the overall evolutionary process. ONE-NAS has been evaluated on real-world datasets including wind turbine sensor data and financial time series, demonstrating superior performance compared to classical TSF methods, online LSTM/GRU networks, and online ARIMA approaches.

![ONE-NAS Architecture](images/onenas.png)

# Selected Publications

1. Zimeng Lyu, Alexander Ororbia, Travis Desell. **"Online Evolutionary Neural Architecture Search for Multivariate Non-Stationary Time Series Forecasting,"** Applied Soft Computing, 2023. (IF: 8.7)

2. Zimeng Lyu, Travis Desell. **"ONE-NAS: An Online NeuroEvolution based Neural Architecture Search for Time Series Forecasting,"** GECCO 2022.


# Getting Started and Prerequisites

ONENAS has been developed to compile using CMake. To use the MPI version, a version of MPI (such as OpenMPI) should be installed.

## OSX Setup
```bash
brew install cmake
brew install mysql
brew install open-mpi
brew install libtiff
brew install libpng
brew install clang-format
xcode-select --install
```

## Building
```bash
mkdir build
cd build
cmake ..
make
```

# Running ONENAS

ONENAS can be run in two different modes - MPI (distributed) or multithreaded. For quick start with example datasets using default settings:

## MPI Version
```bash
# In the root directory:
sh scripts/one-nas/coal_mpi.sh
```

---
© 2025 All Rights Reserved.

