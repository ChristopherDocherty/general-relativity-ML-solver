# Einsteins Field Equations Machine Learning Solver ![](https://img.shields.io/badge/python-3.7.3-blue.svg)



This repository contains an implementation of the [PINN](https://arxiv.org/abs/2006.08472) (Physics Informed Neural Network) algorithm which is used to solve Einstein's field equations - the equation underlying all of General Relativity.


The system that is being solved in this repo is a star of constant density, with the goal being to train a neural network capable of approximating the metric tensor's $g_{rr}$ element.

This approach was found to work well for matching the Christoffel tensor only (which amounts to solving a 1st order differential system of 10 linked equations) but numerical noise from floating point errors prevents the algorithm from successfully solving Einstein's Field Equations. 


