# Gene definitions for NEAT Genomes
# WHY: Defines the basic building blocks (nodes and connections) of a genome.
# RELEVANT FILES: hyperneat/genome.py

import random
from copy import deepcopy

class NodeGene:
    def __init__(self, id, type, activation='sigmoid', bias=0.0):
        self.id = id
        self.type = type  # 'INPUT', 'HIDDEN', 'OUTPUT'
        self.activation = activation
        self.bias = bias
    
    def copy(self):
        return NodeGene(self.id, self.type, self.activation, self.bias)

    def __repr__(self):
        return f"NodeGene(id={self.id}, type={self.type}, bias={self.bias:.3f}, act={self.activation})"

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation, gating_node=None, activation='identity', activation_params=None):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
        self.gating_node = gating_node
        self.activation = activation
        self.activation_params = activation_params if activation_params is not None else []

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation, self.gating_node, self.activation, list(self.activation_params))

    def __repr__(self):
        gate_str = f", gate={self.gating_node}" if self.gating_node is not None else ""
        act_str = f", act={self.activation}" if self.activation != 'identity' else ""
        return f"ConnectionGene(in={self.in_node}, out={self.out_node}, w={self.weight:.3f}, en={self.enabled}, innov={self.innovation}{gate_str}{act_str})"
