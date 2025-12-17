# Population Manager
# WHY: Manages the lifecycle of evolution.
# RELEVANT FILES: hyperneat/species.py, hyperneat/genome.py

import math
import random
from hyperneat.genome import Genome, InnovationCounter
from hyperneat.species import Species

class Population:
    def __init__(self, population_size, input_size, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.innovation_counter = InnovationCounter()
        self.generation = 0
        self.species = []
        self.species_indexer = 0
        
        # Initialize population
        self.population = {} # id -> Genome
        for i in range(population_size):
            g = Genome(i)
            g.configure_new(input_size, output_size, self.innovation_counter)
            self.population[i] = g
            
        self.best_genome = None
            
    def speciate(self, threshold=3.0):
        # Clear species members
        for s in self.species:
            s.reset()
            
        # Assign genomes to species
        for genome in self.population.values():
            found = False
            for s in self.species:
                dist = genome.distance(s.representative)
                if dist < threshold:
                    s.add_member(genome)
                    found = True
                    break
            
            if not found:
                # Create new species
                self.species_indexer += 1
                new_species = Species(self.species_indexer, genome)
                # genome is already added in Species input/init? No, init sets rep and adds it.
                # My species init adds rep to members.
                self.species.append(new_species)
                
        # Remove empty species
        self.species = [s for s in self.species if s.members]

    def run(self, fitness_function, n_generations, fitness_threshold=None):
        stagnation_counter = 0
        last_best_fitness = -float('inf')

        for gen in range(n_generations):
            self.generation += 1
            print(f"Generation {self.generation}")
            
            # 1. Evaluate Fitness
            fitness_function(list(self.population.values()))
            
            # Track best
            curr_best = max(self.population.values(), key=lambda g: g.fitness)
            if self.best_genome is None or curr_best.fitness > self.best_genome.fitness:
                self.best_genome = curr_best.copy()
            
            if curr_best.fitness > last_best_fitness + 0.001: 
                # Improvement found
                last_best_fitness = curr_best.fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            print(f"  Best Fitness: {curr_best.fitness:.4f} (Stagnation: {stagnation_counter})")
            
            if fitness_threshold is not None and curr_best.fitness >= fitness_threshold:
                print("  Solved!")
                return self.best_genome

            # Dynamic Stagnation Handling
            mutation_multiplier = 1.0
            if stagnation_counter > 20:
                print("  STAGNATION DETECTED: Boosting mutation rates!")
                mutation_multiplier = 3.0
            
            if stagnation_counter > 50:
                print("  CRITICAL STAGNATION: Mass Extinction Event initiated.")
                # Reset Population except for top 2
                self.perform_mass_extinction(curr_best)
                stagnation_counter = 0 # Reset counter after extinction
                mutation_multiplier = 1.0 

            # 2. Speciate
            self.speciate()
            
            # 3. Calculate Spawn Amounts
            # Adjust fitness
            total_adjusted_fitness = 0.0
            for s in self.species:
                s.update_staleness()
                s.calculate_adjusted_fitness()
                total_adjusted_fitness += s.adjusted_fitness
            
            if total_adjusted_fitness == 0:
                print("  Total adjusted fitness is 0, resetting...")
                total_adjusted_fitness = 1.0 # Hack
            
            # 4. Reproduction
            new_population = {}
            new_pop_list = []
            
            # Elitism: Keep the very best one across all species
            new_pop_list.append(curr_best.copy())
            
            for s in self.species:
                # Kill stale species
                if s.staleness > 15 and len(self.species) > 1:
                     continue
                     
                # Calculating offspring for this species
                offspring_count = int((s.adjusted_fitness / total_adjusted_fitness) * self.population_size)
                
                # Sort members by fitness
                s.members.sort(key=lambda g: g.fitness, reverse=True)
                
                # Elitism inside species: Keep best one if species is large enough
                # if len(s.members) > 5:
                #    new_pop_list.append(s.members[0].copy())
                #    offspring_count -= 1
                
                if offspring_count <= 0:
                    continue
                    
                # Cull bottom half
                survivors = s.members[:max(1, len(s.members)//2)]
                
                for _ in range(offspring_count):
                    # Select 2 parents
                    p1 = random.choice(survivors)
                    p2 = random.choice(survivors)
                    
                    child = Genome.crossover(p1, p2)
                    child.mutate(self.innovation_counter, mutation_multiplier)
                    new_pop_list.append(child)
            
            # Fill remaining slots if rounding errors
            while len(new_pop_list) < self.population_size:
                # Just breed from random species
                if not self.species: # Should not happen unless all died
                    # Re-seed
                    print("  Extinction event! Re-seeding.")
                    g = Genome(0)
                    g.configure_new(self.input_size, self.output_size, self.innovation_counter)
                    new_pop_list.append(g)
                    continue

                s = random.choice(self.species)
                survivors = s.members[:max(1, len(s.members)//2)]
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = Genome.crossover(p1, p2)
                child.mutate(self.innovation_counter, mutation_multiplier)
                new_pop_list.append(child)
                
            # Limit to population size
            new_pop_list = new_pop_list[:self.population_size]
            
            # Assign keys and replace
            for i, g in enumerate(new_pop_list):
                g.key = i
                self.population[i] = g
                
        return self.best_genome

    def perform_mass_extinction(self, survivor):
        print("  Wiping population except the Elite...")
        self.species = [] # Clear species
        # Re-initialize population based on survivor mutations or fresh starts?
        # Usually keeping survivor + offspring of survivor is good strategies (Delta Coding)
        # Or Just keep survivor and randos.
        
        self.population = {}
        # Keep 1 elite
        self.population[0] = survivor.copy()
        
        # Fill rest with heavy mutated versions of survivor
        for i in range(1, self.population_size):
            g = survivor.copy()
            # Heavy mutation
            g.mutate(self.innovation_counter, rate_multiplier=5.0) 
            g.key = i
            self.population[i] = g
