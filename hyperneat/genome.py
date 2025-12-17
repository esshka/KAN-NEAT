# Genome class for NEAT
# WHY: Encapsulates the genotype (nodes and connections) and evolutionary operators.
# RELEVANT FILES: hyperneat/genes.py, hyperneat/activations.py

import random
from hyperneat.genes import NodeGene, ConnectionGene
from hyperneat.activations import registry as activation_registry

class Genome:
    def __init__(self, key):
        self.key = key
        self.nodes = {} # id -> NodeGene
        self.connections = {} # (in, out) -> ConnectionGene
        self.fitness = 0.0
        # Deep HyperNEAT: Dynamic Substrate Config
        # List of integers, each representing number of nodes in a hidden layer
        self.substrate_config = {'hidden_layers': []}

    def configure_new(self, num_inputs, num_outputs, innovation_counter):
        # Create input nodes
        for i in range(num_inputs):
            # Input nodes usually have identity activation or just pass through
            self.nodes[i] = NodeGene(i, 'INPUT', activation='identity')
            
        # Create output nodes
        # Use lists for keys to avoid runtime modification issues if iterating
        input_keys = list(range(num_inputs))
        output_keys = []
        
        for i in range(num_outputs):
            idx = num_inputs + i
            # Output nodes defaults to identity for KAN (summation nodes)
            self.nodes[idx] = NodeGene(idx, 'OUTPUT', activation='identity')
            output_keys.append(idx)
            
        # Fully connect inputs to outputs
        for in_id in input_keys:
            for out_id in output_keys:
                weight = random.gauss(0, 1.0)
                weight = random.gauss(0, 1.0)
                innov = innovation_counter.get_innovation(in_id, out_id)
                # Force B-Spline for Pure KAN
                act = 'bspline'
                params = [random.gauss(0,1) for _ in range(10)]
                    
                self.connections[(in_id, out_id)] = ConnectionGene(in_id, out_id, weight, True, innov, activation=act, activation_params=params)


    def mutate(self, global_innovation_counter, rate_multiplier=1.0):
        """Randomly apply mutations."""
        if random.random() < 0.8 * rate_multiplier:
            self.mutate_weights(rate_multiplier)
            
        if random.random() < 0.1 * rate_multiplier: 
            self.mutate_add_node(global_innovation_counter)
            
        if random.random() < 0.1 * rate_multiplier: 
            self.mutate_add_connection(global_innovation_counter, allow_recurrence=True)
            
        if random.random() < 0.1 * rate_multiplier: 
            self.mutate_activation()
        
        # New Mutations
        if random.random() < 0.05 * rate_multiplier: 
            self.mutate_add_gate()
            
        if random.random() < 0.02 * rate_multiplier:
            self.mutate_remove_gate()
            
        if random.random() < 0.1 * rate_multiplier: 
            self.mutate_add_recurrent(global_innovation_counter)
            
        if random.random() < 0.02 * rate_multiplier:
            self.mutate_remove_recurrent()
            
        # Deep HyperNEAT Mutations
        if random.random() < 0.02 * rate_multiplier:
            self.mutate_add_substrate_layer()
            
        if random.random() < 0.02 * rate_multiplier:
             self.mutate_remove_substrate_layer()
             
        if random.random() < 0.05 * rate_multiplier:
              self.mutate_increase_layer_density()
              
        if random.random() < 0.05 * rate_multiplier:
             self.mutate_decrease_layer_density()

    def mutate_weights(self, rate_multiplier=1.0):
        for conn in self.connections.values():
            if random.random() < 0.1 * rate_multiplier:
                # Perturb weight
                conn.weight += random.gauss(0, 0.5)
            elif random.random() < 0.1 * rate_multiplier:
                # Replace weight
                conn.weight = random.gauss(0, 1.0)
            
            # Clamp weight to reasonable bounds for stability
            conn.weight = max(-30.0, min(30.0, conn.weight))
            
            # Mutate params (e.g. spline control points)
            if conn.activation_params:
                for i in range(len(conn.activation_params)):
                    if random.random() < 0.1 * rate_multiplier:
                         conn.activation_params[i] += random.gauss(0, 0.5)

        # Mutate biases (Nodes still have biases in KAN, usually)
        for node in self.nodes.values():
            if random.random() < 0.1 * rate_multiplier:
                node.bias += random.gauss(0, 0.5)
                
    def mutate_activation(self):
        # In Pure-Spline KAN, we don't switch types, we only reset params.
        # So this mutation effectively just "perturbs" or "resets" the spline.
        if not self.connections: return
        
        target_conn = random.choice(list(self.connections.values()))
        # Reset params to new random values
        target_conn.activation_params = [random.gauss(0,1) for _ in range(10)]

    def mutate_add_node(self, innovation_counter):
        if not self.connections:
            return
            
        # Pick a random enabled connection to split
        enabled_conns = [c for c in self.connections.values() if c.enabled]
        if not enabled_conns:
            return
            
        conn_to_split = random.choice(enabled_conns)
        conn_to_split.enabled = False
        
        new_node_id = max(self.nodes.keys()) + 1
        # KAN Nodes are summation nodes (identity)
        new_node = NodeGene(new_node_id, 'HIDDEN', activation='identity')
        self.nodes[new_node_id] = new_node
        
        # Add two new connections
        # 1. In -> New (weight 1.0)
        innov1 = innovation_counter.get_innovation(conn_to_split.in_node, new_node_id)
        new_conn1 = ConnectionGene(conn_to_split.in_node, new_node_id, 1.0, True, innov1)
        self.connections[(conn_to_split.in_node, new_node_id)] = new_conn1
        
        # 2. New -> Out (weight = old weight, activation = old activation)
        innov2 = innovation_counter.get_innovation(new_node_id, conn_to_split.out_node)
        new_conn2 = ConnectionGene(new_node_id, conn_to_split.out_node, conn_to_split.weight, True, innov2, activation=conn_to_split.activation, activation_params=list(conn_to_split.activation_params))
        self.connections[(new_node_id, conn_to_split.out_node)] = new_conn2

    def mutate_add_connection(self, innovation_counter, allow_recurrence=True):
        # Pick two random nodes
        node_keys = list(self.nodes.keys())
        input_keys = [k for k, v in self.nodes.items() if v.type in ('INPUT', 'HIDDEN')]
        output_keys = [k for k, v in self.nodes.items() if v.type in ('HIDDEN', 'OUTPUT')]
        
        if not input_keys or not output_keys:
            return

        in_node_id = random.choice(input_keys)
        out_node_id = random.choice(output_keys)
        
        # Don't connect if connection already exists
        if (in_node_id, out_node_id) in self.connections:
            return
            
        # Cycle check
        if not allow_recurrence:
            if self.creates_cycle(in_node_id, out_node_id):
                return


        weight = random.gauss(0, 1.0)
        innov = innovation_counter.get_innovation(in_node_id, out_node_id)
        
        act = 'bspline'
        params = [random.gauss(0,1) for _ in range(10)]
             
        new_conn = ConnectionGene(in_node_id, out_node_id, weight, True, innov, activation=act, activation_params=params)
        self.connections[(in_node_id, out_node_id)] = new_conn

    def mutate_add_gate(self):
        # Pick a random enabled connection to gate
        enabled_conns = [c for c in self.connections.values() if c.enabled and c.gating_node is None]
        if not enabled_conns:
            return
            
        target_conn = random.choice(enabled_conns)
        
        # Pick a random node to act as gate (Inputs, Hidden, Output - though usually input/hidden gate)
        # Any node can gate
        possible_gates = list(self.nodes.values())
        if not possible_gates: return
        
        gate_node = random.choice(possible_gates)
        
        target_conn.gating_node = gate_node.id
        
    def mutate_remove_gate(self):
        # Pick random connection with gate
        gated_conns = [c for c in self.connections.values() if c.gating_node is not None]
        if not gated_conns:
            return
            
        target_conn = random.choice(gated_conns)
        target_conn.gating_node = None
        
    def mutate_add_recurrent(self, innovation_counter):
        # Explicitly try to add a recurrent connection (forms a cycle)
        # 1. Pick two nodes such that creates_cycle is TRUE
        # Or pick src, then find dst that creates cycle (upstream node)
        
        node_keys = list(self.nodes.keys())
        random.shuffle(node_keys)
        
        for src in node_keys:
            for dst in node_keys:
                if (src, dst) in self.connections:
                    continue
                
                # Check if it creates cycle
                if self.creates_cycle(src, dst):
                    # Found one! Add it.
                    weight = random.gauss(0, 1.0)
                    innov = innovation_counter.get_innovation(src, dst)
                    
                    act = 'bspline'
                    params = [random.gauss(0,1) for _ in range(10)]
                    
                    new_conn = ConnectionGene(src, dst, weight, True, innov, activation=act, activation_params=params)
                    self.connections[(src, dst)] = new_conn
                    return # Add only one
                    
    def mutate_remove_recurrent(self):
        # Find cyclic connections and remove one
        # A connection is "cyclic" if it's part of a cycle. 
        # Heuristic: Remove self-loops first, then check others.
         
        # Self loops
        self_loops = [c for c in self.connections.values() if c.in_node == c.out_node and c.enabled]
        if self_loops:
            target = random.choice(self_loops)
            target.enabled = False
            return
             
        # General cycle
        # Iterate and check? Expensive check for mutation. 
        # Let's skip general cycle removal for now or check back-prop
        pass

    def mutate_add_substrate_layer(self):
        # Adds a new layer with random density (e.g. 1-10 nodes)
        # Add at random position
        if len(self.substrate_config['hidden_layers']) >= 5: # Max 5 hidden layers
            return
            
        new_layer_size = random.randint(2, 5)
        if not self.substrate_config['hidden_layers']:
            self.substrate_config['hidden_layers'].append(new_layer_size)
        else:
            idx = random.randint(0, len(self.substrate_config['hidden_layers']))
            self.substrate_config['hidden_layers'].insert(idx, new_layer_size)
            
    def mutate_remove_substrate_layer(self):
        if not self.substrate_config['hidden_layers']:
            return
        
        idx = random.randint(0, len(self.substrate_config['hidden_layers']) - 1)
        self.substrate_config['hidden_layers'].pop(idx)
        
    def mutate_increase_layer_density(self):
        if not self.substrate_config['hidden_layers']:
            return
            
        idx = random.randint(0, len(self.substrate_config['hidden_layers']) - 1)
        # Cap size at 10
        if self.substrate_config['hidden_layers'][idx] < 20: 
             self.substrate_config['hidden_layers'][idx] += 1
             
    def mutate_decrease_layer_density(self):
         if not self.substrate_config['hidden_layers']:
             return
             
         idx = random.randint(0, len(self.substrate_config['hidden_layers']) - 1)
         if self.substrate_config['hidden_layers'][idx] > 1:
               self.substrate_config['hidden_layers'][idx] -= 1
         else:
             # Remove layer if 1 -> 0?
             self.substrate_config['hidden_layers'].pop(idx)

    def creates_cycle(self, in_id, out_id):
        if in_id == out_id: return True
        visited = set()
        stack = [out_id]
        while stack:
            curr = stack.pop()
            if curr == in_id:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            # Find all nodes that 'curr' connects to
            for (src, dst), conn in self.connections.items():
                if src == curr:
                    stack.append(dst)
        return False

    def distance(self, other):
        # Compatibility distance
        # d = c1*E/N + c2*D/N + c3*W
        # E = excess, D = disjoint, W = avg weight diff
        c1, c2, c3 = 1.0, 1.0, 0.4
        
        # Get innovation numbers
        innovs1 = {c.innovation: c for c in self.connections.values()}
        innovs2 = {c.innovation: c for c in other.connections.values()}
        
        all_innovs = set(innovs1.keys()) | set(innovs2.keys())
        
        disjoint = 0
        excess = 0
        weight_diff = 0
        matching = 0
        
        max_innov1 = max(innovs1.keys()) if innovs1 else 0
        max_innov2 = max(innovs2.keys()) if innovs2 else 0
        
        for innov in all_innovs:
            in1 = innov in innovs1
            in2 = innov in innovs2
            
            if in1 and in2:
                matching += 1
                weight_diff += abs(innovs1[innov].weight - innovs2[innov].weight)
            elif in1:
                if innov > max_innov2: excess += 1
                else: disjoint += 1
            elif in2:
                if innov > max_innov1: excess += 1
                else: disjoint += 1
                
        N = max(len(innovs1), len(innovs2))
        if N < 20: N = 1 # Normalizing for small genomes can be weird, commonly ignored for small N
        
        dist = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * (weight_diff / matching if matching > 0 else 0))
        return dist

    def copy(self):
        new_genome = Genome(self.key)
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        if self.substrate_config:
            new_genome.substrate_config = self.substrate_config.copy()
            
        return new_genome

    @staticmethod
    def crossover(genome1, genome2):
        # genome1 must be the fitter parent
        if genome2.fitness > genome1.fitness:
            genome1, genome2 = genome2, genome1
            
        child = Genome(None) # Key will be assigned by population
        
        # Inherit Substrate Config from fitter parent
        if hasattr(genome1, 'substrate_config'):
             child.substrate_config = genome1.substrate_config.copy()
        
        # Inherit all nodes from fitter parent
        # Note: In standard NEAT, we might add extra nodes from disjoint genes of parent2?
        # A simpler approach: Inherit nodes that are part of the inherited connections, 
        # plus maybe unused ones?
        # Safe bet: Copy all nodes from fitter parent.
        child.nodes = {k: v.copy() for k, v in genome1.nodes.items()}
        
        # Crossover connections
        innovs1 = {c.innovation: c for c in genome1.connections.values()}
        innovs2 = {c.innovation: c for c in genome2.connections.values()}
        
        all_innovs = set(innovs1.keys()) | set(innovs2.keys())
        
        for innov in all_innovs:
            in1 = innov in innovs1
            in2 = innov in innovs2
            
            gene_to_add = None
            if in1 and in2:
                # Matching: pick random
                gene_to_add = innovs1[innov] if random.random() < 0.5 else innovs2[innov]
            elif in1:
                # Disjoint/Excess from fitter parent: keep
                gene_to_add = innovs1[innov]
            # else: Disjoint/Excess from weaker parent: discard
            
            if gene_to_add:
                # Ensure nodes exist in child
                if gene_to_add.in_node not in child.nodes:
                    # If the node was in parent2 but not parent1 (fitter), strictly we shouldn't have selected this gene?
                    # But if we selected a gene from fitter parent (in1), the node MUST be in parent1, thus in child.
                    # This check is theoretically redundant if we copied all nodes from parent1.
                    pass 
                
                # Copy gene
                child_conn = gene_to_add.copy()
                child.connections[(child_conn.in_node, child_conn.out_node)] = child_conn
                
        return child

class InnovationCounter:
    def __init__(self):
        self.current_innovation = 0
        self.history = {} # (in, out) -> innov
        
    def get_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key in self.history:
            return self.history[key]
        self.current_innovation += 1
        self.history[key] = self.current_innovation
        return self.current_innovation
