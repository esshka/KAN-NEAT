# Deep HyperNEAT for Copy Task
# RELEVANT FILES: benchmarks/copy_task/task_utils.py, deep_hyperneat_associative.py, hyperneat/substrate.py

import sys
import os
import time
import random

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork
from hyperneat.substrate import Substrate
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
        # A. Create Dynamic Substrate from Genome
        # Start minimal, let evolution add layers
        substrate = Substrate.create_from_genome(genome, input_coords, output_coords)
        
        # Enable Recurrence + Geometric Gating
        substrate.allow_recurrence = True
        substrate.gating_mode = "geometric"
        
        # FULL Connectivity Query
        # We want to allow the CPPN to explore any possible connection
        # Recurrent connections are critical for Copy Task
        substrate.connections_to_query = []
        for src_id in substrate.all_ids:
            for tgt_id in substrate.all_hidden_and_output_ids:
                # CPPN Input: (x1, y1, x2, y2)
                coords = substrate.id_to_coord[src_id] + substrate.id_to_coord[tgt_id]
                substrate.connections_to_query.append({
                    'input': coords,
                    'src_id': src_id,
                    'tgt_id': tgt_id
                })
        
        # B. Build Network
        cppn = FeedForwardNetwork.create(genome)
        net = substrate.build_network(cppn)
        
        # C. Evaluate
        genome.fitness = task.evaluate_fitness(net, num_examples=5)

def run():
    print("--- Deep HyperNEAT Copy Task Benchmark ---")
    
    # CPPN Outputs:
    # 0: Weight, 1: LEO, 2: Gating, 3: Bias, 4: Activation Type
    out_size = 5
    
    pop = Population(POP_SIZE, input_size=4, output_size=out_size) # 4 inputs to CPPN (x1,y1,x2,y2)
    
    # We must start with logic to add substrate mutations? 
    # Population class handles standard mutations.
    # We need to manually inject substrate mutation logic or ensure it's in Genome.mutate()
    # Genome.mutate calls mutate_add_substrate_layer if configured.
    # Let's ensure our population loop does custom mutations if needed.
    # Actually, Population.run() handles standard logic.
    # But Deep HyperNEAT relies on `mutate_add_substrate_layer`.
    # Standard Genome.mutate() DOES call `mutate_add_substrate_layer` with low prob.
    
    # Custom Run Loop to ensure we can debug/monitor
    # Or just use pop.run, but we need to pass fitness function.
    
    winner = pop.run(eval_fitness, n_generations=MAX_GENS, fitness_threshold=FITNESS_THRESHOLD)
    
    if winner:
        print(f"Winner Fitness: {winner.fitness}")
        
        # Rebuild for visual
        substrate = Substrate.create_from_genome(winner, input_coords, output_coords)
        substrate.allow_recurrence = True
        substrate.gating_mode = "geometric"
        # Rebuild query list (duplicate logic, should wrap in func)
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
