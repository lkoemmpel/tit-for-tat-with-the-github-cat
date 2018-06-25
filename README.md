# tit-for-tat-with-the-github-cat
Code for modeling populations as graphs and simulating a myriad of social games.

The following are the main components of the code:

## Graph structure initialization
This code generates a graph to model a population. 
Nodes represent individuals and edges between nodes indicate the potential to interact. 

## Timestep control/indicator function
This functions determines when nodes of the graph should be deleted or reproduce according to varying probabilities (i.e. a Poisson distribution). 

## Interation of games on the graph
Simulation of nodes playing a social game with the neighbors the indicator function has chosen

## Reproduction
Nodes replicate their strategies according to the following parameters:
* Fitness (calculated according to accumulated payoff
* Node "importance"
* Neighboring edge weights


## Getting Started

Code can be run in Python 3.0

### Prerequisites

No current prerequesites 


### Installing

No current packages to be installed.


## Running the tests

There are no current tests for this code.


## Authors

Ashwin Narayan
Olga Medrano
Laura Koemmpel


## Academic papers inspiring this work

## Acknowledgments


