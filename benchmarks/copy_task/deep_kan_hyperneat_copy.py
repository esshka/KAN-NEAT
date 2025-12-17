# Deep KAN-HyperNEAT for Copy Task
# RELEVANT FILES: benchmarks/copy_task/task_utils.py, deep_kan_hyperneat_associative.py, hyperneat/substrate.py

import sys
import os
import time
import random

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork
from hyperneat.substrate import Substrate, KANSubstrate
from benchmarks.copy_task.task_utils import CopyTask, verify_visual, INPUT_SIZE, OUTPUT_SIZE

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
    task = CopyTask(sequence_length=3, delay_length=5)
    
    for genome in genomes:
        # A. Create Dynamic KAN Substrate
        substrate = Substrate.create_from_genome(genome, input_coords, output_coords, substrate_class=KANSubstrate)
        
        # Enable Recurrence + Geometric Gating
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
        genome.fitness = task.evaluate_fitness(net, num_examples=5)

def run():
    print("--- Deep KAN-HyperNEAT Copy Task Benchmark ---")
    
    # CPPN Outputs for KAN:
    # 0: Weight, 1: LEO, 2: Gating, 3: Bias, 4: Activation Type
    # 5: Edge Activation Type (unused?), 6-9: Spline control points?
    # Actually, KANSubstrate expects 10 outputs usually (from `kan_neat_associative.py`)
    out_size = 10
    
    pop = Population(POP_SIZE, input_size=4, output_size=out_size)
    
    winner = pop.run(eval_fitness, n_generations=MAX_GENS, fitness_threshold=FITNESS_THRESHOLD)
    
    if winner:
        print(f"Winner Fitness: {winner.fitness}")
        
        substrate = Substrate.create_from_genome(winner, input_coords, output_coords, substrate_class=KANSubstrate)
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
