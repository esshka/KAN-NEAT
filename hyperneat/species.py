# Species class
# WHY: Groups similar genomes to protect innovation.
# RELEVANT FILES: hyperneat/population.py, hyperneat/genome.py

import random

class Species:
    def __init__(self, key, representative):
        self.key = key
        self.representative = representative
        self.members = [representative]
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.staleness = 0
        self.max_fitness = 0.0

    def add_member(self, genome):
        self.members.append(genome)

    def calculate_average_fitness(self):
        total = sum(m.fitness for m in self.members)
        self.fitness = total / len(self.members)
    
    def calculate_adjusted_fitness(self):
        # Adjusted fitness = fitness / number of members in species
        # But we do this per genome usually.
        # Here we can just store the sum of adjusted fitnesses to determine offspring count.
        for member in self.members:
            # Shared fitness
            member.adjusted_fitness = member.fitness / len(self.members)
        
        self.adjusted_fitness = sum(m.adjusted_fitness for m in self.members)

    def reset(self):
        # New representative is random member from previous generation (top performer usually better? 
        # Standard: random from previous gen, or keeping the old one if it's still good)
        # NEAT-Python: Uses random member as new representative
        if not self.members:
            return
            
        self.representative = random.choice(self.members)
        self.members = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

    def update_staleness(self):
        current_max = max(m.fitness for m in self.members)
        if current_max > self.max_fitness:
            self.max_fitness = current_max
            self.staleness = 0
        else:
            self.staleness += 1
