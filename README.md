Parallel Genetic Algorithm
==========================

The Genetic Algorithm (GA) is a search heuristic utilized to generate useful 
solutions of optimization and search problems. However, as the performance 
requirement in applying this algorithm in the real world is important, fastening the 
algorithm to complete a job is meaningful. In this project, we tried to parallelize this 
useful algorithm using Pthreads with three methods, namely Single Population Model, 
Island Model and Hybrid Model. The Single Population Model is a shared memory 
model which processes multiple selections in a generation. The Island Model splits 
the entire population to several small pieces and assigns them to several islands to 
execute multiple GAs in parallel. The Hybrid Method integrates the two methods and 
shows advantages of the both. We used the 0-1 Knapsack problem as the 
experimental problem and a benchmark with 10000 packages to test our parallel 
algorithm. Results demonstrated the best speedup using the Hybrid Method when 
running the implementation on a four-core machine.

Check the paper for details

https://github.com/marcyliao/PGA/blob/master/Using%20Parallel%20Genetic%20Algorithm%20to%20solve%200-1%20Knapsack%20Problem%20with%20Pthreads.pdf
