# Deep HyperNEAT for Parentheses Task
# RELEVANT FILES: benchmarks/parentheses/task_utils.py, hyperneat/substrate.py

import sys
import os
import time
import random

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork
from hyperneat.substrate import Substrate
from benchmarks.parentheses.task_utils import ParenthesesTask, verify_visual, INPUT_SIZE, OUTPUT_SIZE

# ==============================================================================
# CONFIG
# ==============================================================================
POP_SIZE = 150
MAX_GENS = 500
FITNESS_THRESHOLD = 9.8 

# ==============================================================================
# SUBSTRATE SETUP
# ==============================================================================
def get_layer_coords(num_nodes, y_val):
    if num_nodes == 1: return [(0.0, y_val)]
    x_step = 2.0 / max(1, num_nodes - 1)
    return [(-1.0 + i * x_step, y_val) for i in range(num_nodes)]

input_coords = get_layer_coords(INPUT_SIZE, -1.0)
output_coords = get_layer_coords(OUTPUT_SIZE, 1.0)

def eval_fitness(genomes):
    task = ParenthesesTask(max_depth=4, sequence_length=10)
    
    for genome in genomes:
        substrate = Substrate.create_from_genome(genome, input_coords, output_coords)
        
        # Enforce Recurrence
        substrate.allow_recurrence = True
        substrate.gating_mode = "geometric"
        
        # FULL Connectivity Query
        substrate.connections_to_query = []
        for src_id in substrate.all_ids:
            for tgt_id in substrate.all_hidden_and_output_ids:
                coords = substrate.id_to_coord[src_id] + substrate.id_to_coord[tgt_id]
                substrate.connections_to_query.append({'input': coords, 'src_id': src_id, 'tgt_id': tgt_id})
        
        # B. Build Network
        cppn = FeedForwardNetwork.create(genome)
        net = substrate.build_network(cppn)
        
        # C. Evaluate
        genome.fitness = task.evaluate_fitness(net, num_examples=50)

def run():
    print("--- Deep HyperNEAT Parentheses Benchmark ---")
    
    out_size = 5 # Weight, LEO, Gating, Bias, Activation
    pop = Population(POP_SIZE, input_size=4, output_size=out_size)
    
    winner = pop.run(eval_fitness, n_generations=MAX_GENS, fitness_threshold=FITNESS_THRESHOLD)
    
    if winner:
        print(f"Winner Fitness: {winner.fitness}")
        
        substrate = Substrate.create_from_genome(winner, input_coords, output_coords)
        substrate.allow_recurrence = True
        substrate.gating_mode = "geometric"
        substrate.connections_to_query = []
        for src_id in substrate.all_ids:
            for tgt_id in substrate.all_hidden_and_output_ids:
                coords = substrate.id_to_coord[src_id] + substrate.id_to_coord[tgt_id]
                substrate.connections_to_query.append({'input': coords, 'src_id': src_id, 'tgt_id': tgt_id})
                
        cppn = FeedForwardNetwork.create(winner)
        net = substrate.build_network(cppn)
        verify_visual(net)

if __name__ == "__main__":
    run()
