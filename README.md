Genetic Algorithm module that I developed for optimization problems. (Presented here for demonstration only)

Module Genetica.py contains two classes and a function.

```py
from Genetica import Population, ProblemDef, runGA
```

`Population` class
* Used to create population with the given population size and a chromosome length. Performs operations of selection, crossover and mutation to generate next generations. 


`ProblemDef` class
* Abstract class from which problem definitions will be inherited. 
* Abstract functions `chromosome_length`, `decode`, `objective_fcn`, `fitness_fcn` are to be implemented for the problem case.

`runGA` function
* Runs the Genetic Algorithm search with the given parameters.
