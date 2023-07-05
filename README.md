Genetic Algorithm module that I developed for optimization problems. (Presented here for demonstration only)

Module Genetica.py contains two classes:


`ProblemDef` class
* Abstract class from which problem definitions will be inherited. 
* Abstract functions `chromosome_length`, `decode`, `objective_fcn`, `fitness_fcn` are to be implemented for the problem case.

`Population` class
* Used to create population with the given population size and a chromosome length. Performs operations of selection, crossover and mutation to generate next generations. 

```py
from Genetica import ProblemDef, Population

class MyOptimizationProblem(ProblemDef)

    # Implement functions for the specific problem:
    # __init__()
    # chromosome_length()
    # decode()
    # objective_fcn()
    # fitness_fcn()

problem = MyOptimizationProblem()
population = Population(problem, pop_size=40) 
population.run(terminal_ofv=0.1) 
```
