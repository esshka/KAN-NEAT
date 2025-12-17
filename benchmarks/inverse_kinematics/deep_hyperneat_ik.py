# Deep HyperNEAT for Inverse Kinematics
# RELEVANT FILES: benchmarks/inverse_kinematics/task_utils.py

import sys
import os
import time
import random

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork
from hyperneat.substrate import Substrate
from benchmarks.inverse_kinematics.task_utils import InverseKinematicsTask, verify_visual, INPUT_SIZE, OUTPUT_SIZE

# ==============================================================================
# CONFIG
# ==============================================================================
POP_SIZE = 150
MAX_GENS = 300
FITNESS_THRESHOLD = 95.0 

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
    task = InverseKinematicsTask()
    
    for genome in genomes:
        substrate = Substrate.create_from_genome(genome, input_coords, output_coords)
        
        # Standard FeedForward setup. No recurrence needed for function approx.
        substrate.allow_recurrence = False
        substrate.gating_mode = "geometric"
        substrate.connections_to_query = []
        for src_id in substrate.all_ids:
            for tgt_id in substrate.all_hidden_and_output_ids:
                if substrate.layer_y[src_id] < substrate.layer_y[tgt_id]: # Feedforward only check
                    coords = substrate.id_to_coord[src_id] + substrate.id_to_coord[tgt_id]
                    substrate.connections_to_query.append({'input': coords, 'src_id': src_id, 'tgt_id': tgt_id})
        
        cppn = FeedForwardNetwork.create(genome)
        net = substrate.build_network(cppn)
        
        genome.fitness = task.evaluate_fitness(net, num_examples=50)

def run():
    print("--- Deep HyperNEAT Inverse Kinematics Benchmark ---")
    
    out_size = 5 
    pop = Population(POP_SIZE, input_size=4, output_size=out_size)
    
    winner = pop.run(eval_fitness, n_generations=MAX_GENS, fitness_threshold=FITNESS_THRESHOLD)
    
    if winner:
        print(f"Winner Fitness: {winner.fitness}")
        
        substrate = Substrate.create_from_genome(winner, input_coords, output_coords)
        substrate.allow_recurrence = False
        substrate.gating_mode = "geometric"
        substrate.connections_to_query = []
        for src_id in substrate.all_ids:
            for tgt_id in substrate.all_hidden_and_output_ids:
                if substrate.layer_y[src_id] < substrate.layer_y[tgt_id]:
                    coords = substrate.id_to_coord[src_id] + substrate.id_to_coord[tgt_id]
                    substrate.connections_to_query.append({'input': coords, 'src_id': src_id, 'tgt_id': tgt_id})
                
        cppn = FeedForwardNetwork.create(winner)
        net = substrate.build_network(cppn)
        verify_visual(net)

if __name__ == "__main__":
    run()
