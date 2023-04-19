# Fast Implementation for Building Approximate KNN Graph

This repository is an open-source code for the [SIGMOD 2023 Programming Contest](http://sigmod2023contest.eastus.cloudapp.azure.com/index.shtml), which challenges participants to design and implement efficient and scalable algorithms for constructing approximate K-NN graphs on large-scale datasets.

# Team

- Team: X2A3008M
- Members:
  Number of members: 1
  | Full Name | Institution | Email |
  |-----------|--------------------|------------------------|
  | Meng Chen | Fudan University | mengchen22@m.fudan.edu.cn |

# Solution Overview

The solution is based on an efficient algorithm called [NN-Descent](https://www.cs.princeton.edu/cass/papers/www11.pdf), which iteratively updates the K-NN graph by exploring local neighborhoods. The code is forked off from [kgraph](https://github.com/aaalgo/kgraph), a library for fast approximate nearest neighbor search.

The implementation extends and optimizes the original kgraph code in several aspects, such as Improving the algorithm by using a better neighbor selection strategy, enhancing the memory efficiency, Boosting the computation efficiency by using parallelism, vectorization, and cache optimization techniques.

The solution achieves performance improvement over the original algorithm and code, as well as other methods for approximate K-NN graph construction like EFANNA.

> note: Due to the time limitation of the contest, the current implementation is only optimized and adapted for the graph BUILD phase of the contest. The graph QUERY phase,is not yet optimized and may need further improvement.

## Dependencies

HardWare:
intel cpu with skylake architecture. (newer is better, but compile option **should be changed** in CMake appropriately, or it will be significant slow).

- [mimalloc](https://github.com/microsoft/mimalloc) open-source allocator from microsoft
- Boost >= 1.65
- CMake >= 3.2

## Build & Run

Here we upload `dummy-data.bin` from contest homepage as example.

### Build

```
make clean
cmake . -DCMAKE_BUILD_TYPE=Release  && make -j
```

### Run

```
./index --data ./dummy-data.bin --output output.bin -K 100 -L 165 \
-S 85 -R 400 --iterations 6 --raw --dim 100 --skip 4
```

The options above could be used to reproduce a recall level of 0.981 in a 10M dataset within 30min on an Azure Machine provided by the contest organizers.
