
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
        """ Return objective function value as a minimization problem """
        pass
    
    @abstractmethod
    def fitness_fcn(self):
        pass
  
    
class Population():
    """ Population class for
    + Creating members 
    + Evaluating fitness values by calling problem definition object methods.
    + Generating new members using random selection, crossover and mutation.
    
    Methods
    -------
    run(terminal_ofv, max_generation=10000, elite=False)
        Run GA optimization
    
    """
    
    def __init__(self, problem, pop_size=30, resume=False, 
                 file='population.pkl', clear_hist=False):
        """
        Constructor for Population class.    

        Parameters
        ----------
        problem : ProblemDef
            Instance of problem definition class inherited from abstract ProblemDef.
            
        pop_size : int, optional
            Population size (default is 30)
            
        resume : bool, optional
            If True, the  run starts from the pickled population. 
            Otherwise, a new population is created. (default is False)
            
        file : str, optional
            File name for resuming the run. (default: 'population.pkl')
            Ignored if resume = False.
            
        clear_hist : bool, optional
            Determines if the list of best scores will be cleared or not on resume.
            (default is False) Ignored if resume = False.
        """
        
        self.problem = problem
        self.chr_len = problem.chromosome_length()
        self.members = []
        self.fitnesses = []
        self.best_ofv = np.Inf
        self.ofv_hist = []
        self.MUT_RATIO = 0.0030
        
        if resume:
            print("Loading existing population...")
            self.resume_population(file) 
            if clear_hist:
                self.ofv_hist = []
        else:
            print("Initializing population...")
            self.create_random_members(pop_size)    
        
        
    def create_random_members(self, pop_size):
        """
        Initializes population with random members.
        
        Parameters
        ----------
        pop_size : int 
            Number of members in the population.
        """
        
        for i in range(pop_size):           
            chromosome = np.round(np.random.uniform(size=self.chr_len)).astype('uint8')
            self.members.append(chromosome)
    
    
    def calc_fitness(self):
        """
        Calculates fitness values for each population member.
        Sets self.fitnesses attribute.
        """
        
        obj_fcn_vals = np.zeros( len(self.members) )
        for i, seq in enumerate(self.members):
            ofv = self.problem.objective_fcn(seq)
            obj_fcn_vals[i] = ofv
            
            if ofv < self.best_ofv:
                self.best_ofv = ofv
                self.best_seq = seq.copy()
        
        self.pop_best_ofv = obj_fcn_vals.max()
        self.ofv_hist.append(self.pop_best_ofv)
              
        self.fitnesses = self.problem.fitness_fcn(obj_fcn_vals)
        

    def create_new_generation(self, elite):
        """
        Creates new population members by,
        1. Selection according to fitness values
        2. Creating offsprings by applying two-point crossover to parents
        3. Applying mutation
        
        Parameters
        ----------
        elite : bool
            Determines if elitist strategy is applied or not.

        """
        # Crossover
        new_members = []
        # For each pair, select with fitness
        for k in range(len(self.fitnesses)//2):
            
            # Select with fitness ---
            i1 = self.select()
            i2 = self.select()
            while (i1==i2):
                i2 = self.select()          

            off1, off2 = self.crossover(self.members[i1], self.members[i2])
            new_members.append(off1)
            new_members.append(off2)
        
        # Mutation
        for sequence in new_members:
            self.mutate(sequence)         
        
        self.members = new_members
        if elite:
            self.members[0] = self.best_seq.copy()
        self.fitnesses = []              
                
         
    def select(self):
        """
        Selects a parent for crossover with propabilites proportional to the 
        fitness values.
        """
        cum_f = np.array(self.fitnesses).cumsum()
        r = np.random.uniform(0, cum_f[-1])
        selection_index = np.where(cum_f >= r)[0][0]
        return selection_index
    
    
    def crossover(self, parent1, parent2):
        """ Create offsprings with two-point crossover on parents """
        
        c1, c2 = self.crossover_loc()        
   
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
    
        
    def resume_population(self, file):
        """ Load pickled poplation """
        
        file = open(file, 'rb')
        D = pickle.load(file)
        self.members = D["members"]
        self.best_ofv =  -np.Inf 
        self.best_seq = D["best_seq"] 
        self.ofv_hist = D["ofv_hist"]
            
        # Inject best sequence to the population
        self.members[0] = self.best_seq.copy()
              
    def pickle_population(self):
        """ Save population to pickle file """
        
        D = {"members": self.members,
             "best_ofv": self.best_ofv,
             "best_seq": self.best_seq,
             "ofv_hist": self.ofv_hist}
        file = open('population.pkl', 'wb')
        pickle.dump(D, file)
        
          
    def run(self, terminal_ofv, max_generation=10000, elite=False):
        """ Run GA optimization. 
        Every 10th population is saved to the default pickle file.
        
        Parameters
        ----------
        terminal_ofv : float
            Objective function value (score) to be reached for 
            terminating the run.
            
        max_generation : int, optional
            Maximum number of generations for the run. (default=10000)
            
        elite : bool, optional
            Determines if elitist strategy will be used. If True, the 
            best member is directly transfered to the next generation.                        
            
        """
        
        st = "Gen.:{0:6d} > Pop. best: {1:7.4f}  Overall best: {2:7.4f} "
        def print_line(generation):
            print_str = st.format( 
                generation, 
                self.pop_best_ofv,
                self.best_ofv)
            
            print(print_str)  #, end="\r"
        
        print("Running generations...")
        generation = 0
        self.calc_fitness()
        print_line(generation)
        
        while (generation <= max_generation):
            generation += 1        
            self.create_new_generation(elite)        
            self.calc_fitness()
            print_line(generation)
            
            if self.best_ofv <= terminal_ofv:            
                break
                
            if (generation % 10) == 9:
                self.pickle_population()
        
        if self.best_ofv <= terminal_ofv:
            print("\nRun ended reaching terminal objective function value.")
        else:
            print("\nObjective function target could not be reached at maximum number of generations.")
            
        self.pickle_population()