# Standard Recurrent NEAT for Parentheses Task
# RELEVANT FILES: benchmarks/parentheses/task_utils.py

import sys
import os
import time
import random

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyperneat.genome import Genome, InnovationCounter
from hyperneat.genes import NodeGene, ConnectionGene
from hyperneat.phenotype import RecurrentNetwork
from hyperneat.species import Species
from benchmarks.parentheses.task_utils import ParenthesesTask, verify_visual, INPUT_SIZE, OUTPUT_SIZE

# ==============================================================================
# CONFIG
# ==============================================================================
POP_SIZE = 150
MAX_GENS = 500
FITNESS_THRESHOLD = 9.8 

# ==============================================================================
# GENOME & POPULATION
# ==============================================================================

def create_genome(key, num_inputs, num_outputs, innov):
    genome = Genome(key)
    # Inputs
    for i in range(num_inputs):
        genome.nodes[i] = NodeGene(i, 'INPUT', activation='identity')
    # Outputs
    out_keys = []
    for i in range(num_outputs):
        idx = num_inputs + i
        genome.nodes[idx] = NodeGene(idx, 'OUTPUT', activation='sigmoid')
        out_keys.append(idx)
        
    # Start Fully Connected but minimal weights
    in_keys = list(range(num_inputs))
    for i in in_keys:
        for o in out_keys:
            w = random.gauss(0, 0.5)
            inv = innov.get_innovation(i, o)
            conn = ConnectionGene(i, o, w, True, inv, activation='identity')
            genome.connections[(i,o)] = conn
            
    return genome

class NEATPopulation:
    def __init__(self, size):
        self.size = size
        self.innov = InnovationCounter()
        self.species_list = []
        self.species_idx = 0
        self.pop = {}
        for i in range(size):
            self.pop[i] = create_genome(i, INPUT_SIZE, OUTPUT_SIZE, self.innov)
        self.best_genome = None
        self.generation = 0
        
    def speciate(self):
        for s in self.species_list: s.reset()
        for g in self.pop.values():
            found = False
            for s in self.species_list:
                if g.distance(s.representative) < 3.0:
                    s.add_member(g)
                    found = True
                    break
            if not found:
                self.species_idx += 1
                self.species_list.append(Species(self.species_idx, g))
        self.species_list = [s for s in self.species_list if s.members]

    def run(self):
        task = ParenthesesTask(max_depth=4, sequence_length=10)
        
        for gen in range(MAX_GENS):
            self.generation = gen + 1
            
            # Evaluate
            best_fit = 0.0
            best_g = None
            
            for g in self.pop.values():
                net = RecurrentNetwork.create(g)
                g.fitness = task.evaluate_fitness(net, num_examples=50) # Need more samples for stability
                if g.fitness > best_fit:
                    best_fit = g.fitness
                    best_g = g
            
            if best_g:
                if self.best_genome is None or best_g.fitness > self.best_genome.fitness:
                    self.best_genome = best_g.copy()
            
            print(f"Gen {self.generation} | Best Fit: {best_fit:.4f} | Species: {len(self.species_list)}")
            
            if best_fit >= FITNESS_THRESHOLD:
                print("SOLVED!")
                return self.best_genome
            
            # Reproduction
            self.speciate()
            new_pop = {}
            new_pop[0] = self.best_genome.copy()
            new_pop[0].key = 0
            
            idx = 1
            while idx < self.size:
                s = random.choice(self.species_list)
                if not s.members: continue
                
                parent = random.choice(s.members)
                child = parent.copy()
                child.mutate(self.innov, rate_multiplier=1.0)
                
                # Recurrence is crucial for counting!
                if random.random() < 0.2: # Higher prob for recurrence
                    child.mutate_add_recurrent(self.innov)
                
                child.key = idx
                new_pop[idx] = child
                idx += 1
            self.pop = new_pop

        return self.best_genome

if __name__ == "__main__":
    print("--- NEAT Parentheses Benchmark ---")
    pop = NEATPopulation(POP_SIZE)
    winner = pop.run()
    
    if winner:
        print(f"Winner Fitness: {winner.fitness}")
        net = RecurrentNetwork.create(winner)
        verify_visual(net)
