# HyperNEAT from Scratch (Pure Python)

A clean, dependency-free implementation of **HyperNEAT** (and the underlying **NEAT** algorithm) written in pure Python.

## Features

- **Pure Python**: No external dependencies (no `numpy`, `scikit-learn`, etc.).
- **NEAT Engine**: Full implementation of NeuroEvolution of Augmenting Topologies.
  - Speciation to protect innovation.
  - Complexification (starting minimal and growing).
  - Historical Markings (Innovation Numbers).
- **HyperNEAT Substrate**: Geometric connectivity mapping.
  - Uses CPPNs (Compositional Pattern Producing Networks) to paint weights onto a target network based on coordinates.
- **Customizable**: Easy to extend activations, genes, and mutation logic.

## Project Structure

```
.
├── hyperneat/
│   ├── activations.py   # Activation functions (Sigmoid, Gaussian, Sine, etc.)
│   ├── genes.py         # NodeGene and ConnectionGene
│   ├── genome.py        # Genotype & Evolutionary Operators (Mutate, Crossover)
│   ├── phenotype.py     # FeedForwardNetwork builder
│   ├── population.py    # Evolution loop & Reproduction logic
│   ├── species.py       # Speciation logic
│   └── substrate.py     # HyperNEAT coordinate mapper
├── xor_test.py          # NEAT verification (XOR problem)
└── substrate_test.py    # HyperNEAT substrate verification
```

## Usage

### 1. Standard NEAT (Evolving a Network)

See `xor_test.py` for a full example.

```python
from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork

def eval_fitness(genomes):
    for genome in genomes:
        # Create phenotype
        net = FeedForwardNetwork.create(genome)
        
        # Evaluate
        output = net.activate([0.0, 1.0])
        error = (output[0] - 1.0) ** 2
        genome.fitness = 4.0 - error

# Initialize Population (size, inputs, outputs)
pop = Population(150, 2, 1)

# Run Evolution
winner = pop.run(eval_fitness, 100)
```

### 2. HyperNEAT (Using Substrate)

See `substrate_test.py` for context.

```python
from hyperneat.substrate import Substrate
from hyperneat.phenotype import FeedForwardNetwork

# Define Geometry
inputs = [(-1, -1), (0, -1), (1, -1)]
outputs = [(-1, 1), (0, 1), (1, 1)]
substrate = Substrate(inputs, outputs)

def eval_hyperneat_fitness(genomes):
    for genome in genomes:
        # 1. Create CPPN from Genome
        cppn = FeedForwardNetwork.create(genome)
        
        # 2. Build Target Network via Substrate
        # Substrate queries CPPN with (x1, y1, x2, y2)
        target_net = substrate.build_network(cppn)
        
        # 3. Evaluate Target Network
        # ... your task specific evaluation ...
```

### 3. Advanced Features

#### Link Expression Output (LEO) / Static Gating
Enable LEO to let the CPPN determine connection existence locally (`use_leo=True`). This requires the CPPN to output 2 values per query: `[weight, gate]`.

```python
# Initialize Substrate with LEO
substrate = Substrate(inputs, outputs, hidden_layers, use_leo=True)

# CPPN must have 2 outputs (Weight, Gate)
pop = Population(150, 4, 2) 
```

#### Dynamic Gating (Neuromodulation)
Allow a node's activation to dynamically modulate a connection's weight: `output = input * weight * gate_node_val`.

```python
# Create Network
net = FeedForwardNetwork.create(genome)

# Apply Gate: Node 5 gates connection from Node 1 to Node 2
# net.gate(gating_node_id, src_id, tgt_id)
net.gate(5, 1, 2)

# During activation, Node 5's value will scale the 1->2 connection
```

#### Visualization
Visualize the substrate geometry and network connectivity, including gating lines.

```python
from hyperneat.visualize import draw_net

draw_net(substrate, net, filename="network_viz.png")
```

#### Recurrent Networks
Enable recurrence in the substrate to allow cycles.

```python
substrate = Substrate(..., allow_recurrence=True)
# Builds a RecurrentNetworkFull instead of FeedForwardNetwork
net = substrate.build_network(cppn)
```

## Configuration

You can tweak mutation rates and speciation thresholds in:
- `hyperneat/genome.py` (Mutation probabilities)
- `hyperneat/population.py` (Speciation threshold)

## KAN-NEAT (Kolmogorov-Arnold NEAT)

This implementation supports **Kolmogorov-Arnold Networks**, where complexity lies in the **edges** (learnable activation functions) rather than the nodes (which are simple sums).

### Features
- **Learnable Edges**: Connections evolve activation functions (e.g., `sin`, `gauss`, `bspline`, `identity`).
- **B-Splines**: Supports evolvable B-Spline connections with learnable control points.
- **Memory**: Recurrent B-Spline loops act as memory cells.

### Example Usage
See `kan_example.py` for solving XOR using KANs.

```python
# KAN behavior is enabled by default in the Genome/Phenotype.
# New connections default to 'bspline'.

from hyperneat.population import Population

def eval_fitness(genomes):
    for genome in genomes:
        # Create KAN Phenotype
        net = FeedForwardNetwork.create(genome)
        # Edges will apply their specific functions (e.g., spline interpolation)
        output = net.activate(inputs) 
        # ...
```
