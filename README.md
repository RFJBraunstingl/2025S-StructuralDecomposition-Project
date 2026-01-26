# 2025S Structural Decomposition and Algorithms

Goal of this repo is to implement an algorithm to calculate the minimum dominating set size of a graph
by utilizing dynamic programming on tree decompositions of graphs.  

The algorithm in use is from chapter 7.3.2 of the book [Parameterized Algorithms](https://www.mimuw.edu.pl/~malcin/book/parameterized-algorithms.pdf) by Cygan et al.

For the algorithm implementation see `min-dominating-set-v2.py`.

## Running
There are samples in `samples`.
```
python min-dominating-set-v2.py samples/C3.gr
python min-dominating-set-v2.py samples/C4.gr
python min-dominating-set-v2.py samples/C5.gr
python min-dominating-set-v2.py samples/C6.gr
python min-dominating-set-v2.py samples/C7.gr
python min-dominating-set-v2.py samples/simple.gr
python min-dominating-set-v2.py samples/simple2.gr
```
