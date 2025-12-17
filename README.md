# HyperNEAT from Scratch (Pure Python)

A clean, dependency-free implementation of **HyperNEAT** (NeuroEvolution of Augmenting Topologies with Hypercube-based Geometric Connectivity) written in pure Python.

This project also includes **KAN-NEAT**, an extension that evolves **Kolmogorov-Arnold Networks** where connections have learnable activation functions (B-Splines, etc.) instead of just scalar weights.

## Features

- **Pure Python**: No external dependencies (no `numpy`, `scikit-learn`, etc.).
- **NEAT Engine**: Full implementation of NeuroEvolution of Augmenting Topologies.
  - Speciation to protect innovation.
  - Complexification (starting minimal and growing).
  - Historical Markings (Global Innovation Numbers).
- **HyperNEAT Substrate**: Geometric connectivity mapping.
  - Uses CPPNs (Compositional Pattern Producing Networks) to "paint" weights onto a target network based on coordinates.
- **KAN-NEAT**: Support for evolving edges with learnable activation functions (B-splines, Sine, Gaussian, etc.).
- **Deep HyperNEAT**: Multi-layer substrate configurations.
- **Memory Support**: Capable of evolving Recurrent Neural Networks (RNNs) with dynamic gating and self-loops.

## Project Structure

```
.
├── hyperneat/           # Core Library
│   ├── activations.py   # Activation functions (Sigmoid, B-Spline, etc.)
│   ├── genes.py         # NodeGene and ConnectionGene (supports KAN edges)
│   ├── genome.py        # Genotype & Evolutionary Operators (Mutate, Crossover)
│   ├── phenotype.py     # FeedForwardNetwork & RecurrentNetwork
│   ├── population.py    # Evolution loop & Reproduction logic
│   ├── species.py       # Speciation logic
│   └── substrate.py     # HyperNEAT coordinate mapper
├── benchmarks/          # Experiments & Performance Tests
│   ├── copy_task/       # Sequential Memory Task
│   ├── inverse_kinematics/ # Geometric Control Task
│   └── parentheses/     # Stack/Recursive Logic Task
└── README.md
```

## Benchmarks

The `benchmarks/` directory contains rigorous tests comparing 4 different algorithm variants. Each benchmark folder typically contains a runner for each variant.

| Benchmark | Description | Goal |
|-----------|-------------|------|
| **Copy Task** | Network must remember a sequence of inputs and replay them after a delay. | Test **Memory** and Recurrence. |
| **Inverse Kinematics** | Calculate joint angles to reach a target (x, y) coordinate. | Test **Geometric Understanding** and continuous function approximation. |
| **Parentheses** | Determine if a string of nested parentheses is valid. | Test **Hierarchical/Stack-like Logic**. |

### Algorithm Variants

1.  **NEAT**: Standard NeuroEvolution of Augmenting Topologies. Direct encoding.
    *   File: `neat_*.py`
2.  **KAN-NEAT**: NEAT with **Learnable Edges**. Connections can be B-Splines or other functions.
    *   File: `kan_neat_*.py`
3.  **Deep HyperNEAT**: HyperNEAT with a deep CPPN. Indirect encoding.
    *   File: `deep_hyperneat_*.py`
4.  **Deep KAN-HyperNEAT**: HyperNEAT where the CPPN itself is a KAN. Indirect encoding with learnable edges in the creating network.
    *   File: `deep_kan_hyperneat_*.py`

## Usage

### Running a Benchmark

To run a benchmark, simply execute the python script from the root directory:

```bash
# Run Standard NEAT on the Copy Task
poetry run python benchmarks/copy_task/neat_copy.py

# Run KAN-HyperNEAT on Inverse Kinematics
poetry run python benchmarks/inverse_kinematics/deep_kan_hyperneat_ik.py
```

### Basic Example (XOR with KAN-NEAT)

You can create simple scripts to test the engine. Here is a conceptual example of how to setup a population:

```python
from hyperneat.population import Population
from hyperneat.phenotype import FeedForwardNetwork

def eval_fitness(genomes):
    for genome in genomes:
        # Create phenotype (supports KAN edges by default)
        net = FeedForwardNetwork.create(genome)
        
        # Evaluate XOR
        fitness = 4.0
        data = [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]
        for inputs, expected in data:
            output = net.activate(inputs)[0]
            fitness -= (output - expected)**2
        genome.fitness = fitness

# Initialize Population (150 individuals, 2 inputs, 1 output)
pop = Population(150, 2, 1)

# Run Evolution
winner = pop.run(eval_fitness, 50)
print(f"Winner Fitness: {winner.fitness}")
```

## Advanced Configuration

### HyperNEAT Substrate
To use HyperNEAT, you define a `Substrate` with coordinates:

```python
from hyperneat.substrate import Substrate

# Define 2D coordinates for inputs and outputs
input_coords = [(-1, -1), (1, -1)]
output_coords = [(0, 1)]

# Create Substrate
substrate = Substrate(input_coords, output_coords)

# Build Network from Genome (CPPN)
target_net = substrate.build_network(cppn_genome)
```

### KAN Nodes vs Edges
In this implementation, **Nodes** are aggregation units (summation), and **Edges** (Connections) contain the non-linearity (activation function).
- To enable KAN-behavior, ensure the genome mutation parameters allow modifying `activation_type` of `ConnectionGene`.
- Supported activations: `identity`, `sigmoid`, `gauss`, `sine`, `bspline` (learnable).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
