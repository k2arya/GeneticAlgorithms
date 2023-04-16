
"""
Module for Genetic Algorithms.

(c) Korhan Kanar
"""

import pickle
from abc import ABC, abstractmethod

import numpy as np

 
class ProblemDef(ABC):
    """ Abstract Problem Definition Class.
    Problem Definition should implement the abstract methods below.
    """
    
    @abstractmethod
    def chromosome_length(self):
        pass
    
    @abstractmethod
    def decode(self):
        pass
    
    @abstractmethod
    def objective_fcn(self):
        """ Return objective function value as a maximization problem """
        pass
    
    @abstractmethod
    def fitness_fcn(self):
        pass
     
                                
class Population():
    """ Population class for
    + Creating members 
    + Evaluating fitness values by calling problem definition classes.
    + Generating new members using random selection, crossover and mutation.
    """
    
    def __init__(self, problem):
        """
        Constructor for Population class.    

        Parameters
        ----------
        problem : instance of problem definition class inherite from 
                  abstract ProblemDef.

        Returns
        -------
        None.
        """
        
        self.problem = problem
        self.chr_len = problem.chromosome_length()
        self.members = []
        self.fitnesses = []
        self.best_ofv = -np.Inf
        self.best = None
        self.MUT_RATIO = 0.0030  
        
        
    def initialize_members(self, pop_size=20):
        """
        Initializes population with random members.
        
        Parameters
        ----------
        pop_size : int (default=20)
                   Size of the population.
        """
        
        for i in range(pop_size):           
            chromosome = np.round(np.random.uniform(size=self.chr_len)).astype('uint8')
            self.members.append(chromosome)
    
    def calc_fitness(self):
        """
        Calculates fitness values for each population member.
        """

        obj_fcn_vals = np.zeros( len(self.members) )
        for i, seq in enumerate(self.members):
            ofv = self.problem.objective_fcn(seq)
            obj_fcn_vals[i] = ofv
            
            if ofv > self.best_ofv:
                self.best_ofv = ofv
                self.best_seq = seq
                #self.save_sequence(seq, cost)
        
        self.pop_best_ofv = obj_fcn_vals.min()               
        
        self.fitnesses = self.problem.fitness_fcn(obj_fcn_vals)
                

    def create_new_generation(self):
        # Crossover
        new_members = []
        # For each pair, select with fitness
        for k in range(len(self.fitnesses)//2):
            
            # Select with fitness ---
            i1 = self.select()
            i2 = self.select()
            while (i1==i2):
                i2 = self.select()
            #print("Crossover indices:", i1,i2)           

            off1, off2 = self.crossover(self.members[i1], self.members[i2])
            new_members.append(off1)
            new_members.append(off2)
        
        # Mutation
        for sequence in new_members:
            self.mutate(sequence)         
        
        self.members = new_members
        self.fitnesses = []              
                
         
    def select(self):
        """
        Selects a parent for crossover according to the 
        fitness values.
        """
        cum_f = np.array(self.fitnesses).cumsum()
        r = np.random.uniform(0, cum_f[-1])
        selection_index = np.where(cum_f >= r)[0][0]
        return selection_index
    
    
    def crossover(self, parent1, parent2):
        
        c1, c2 = self.crossover_loc()        
        #print("Crossover locations:", c1, c2)    
        # Crossover region
        cross = np.array([False]*self.chr_len).astype(bool)
        cross[c1:c2] = True
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()        
        offspring1[cross] = parent2[cross]
        offspring2[cross] = parent1[cross]
        
        return offspring1, offspring2
    
    
    def crossover_loc(self):
        """
        Returns two random location indices for crossover.
        """
        c1 = c2 = int(np.random.uniform(0,1) * self.chr_len)
        while abs(c1-c2) <= 2:
            c2 = int(np.random.uniform(0,1) * self.chr_len)
        return c1, c2
    
    
    def mutate(self, seq):
        """
        Mutates each element in the sequence according to 
        the mutation ratio parameter.
        """
        pr = np.random.uniform(0,1, self.chr_len)
        r0 = (pr < self.MUT_RATIO) & (seq==0)
        r1 = (pr < self.MUT_RATIO) & (seq==1)
        seq[r0] = 1
        seq[r1] = 0 
    
    
    def save_sequence(self, seq, cost):
        fname = f"sequences/{int(cost):07d}.npy"
        np.save(fname, seq)
        
    def resume_population(self):
        file = open('population.pkl', 'rb')
        D = pickle.load(file)
        self.members = D["members"]
              
    def pickle_population(self):
        D = {"members": self.members }
        file = open('population.pkl', 'wb')
        pickle.dump(D, file)




def runGA(population, max_generation=10000, terminal_ofv=0):
    """ Run GA generations. 
    """
    
    population.initialize_members()
    population.calc_fitness()
    
    generation = 1
    st = "Iter:{0:6d} > Pop. best: {1:6.1f}  Overall best: {2:6.1f} "
    
    print("Running generations...")
    while (generation <= max_generation):
                
        population.create_new_generation()        
        population.calc_fitness()
        
        print_str = st.format( 
            generation, 
            population.pop_best_ofv,
            population.best_ofv)
        
        print(print_str, end="\r")
        
        if population.best_ofv >= terminal_ofv:            
            break
        
        generation += 1
    
    if population.best_ofv >= terminal_ofv:
        print("\nRun ended reaching terminal objective function value.")
    else:
        print("\nObjective function target could not be reached at maximum number of generations.")
        
    population.pickle_population()
