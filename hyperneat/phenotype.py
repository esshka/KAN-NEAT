# Phenotype builder
# WHY: Transforms the genetic encoding (Genome) into a functional network.
# RELEVANT FILES: hyperneat/genome.py

from hyperneat.activations import registry as activation_registry

def bspline_eval(x, params):
    # Simple linear interpolation on fixed grid [-3, 3]
    # params: list of values at grid points
    
    # Handle NaN/inf
    import math
    if math.isnan(x) or math.isinf(x):
        return 0.0
    
    n = len(params)
    if n < 2: return x * params[0] if n==1 else 0.0
    
    min_x, max_x = -3.0, 3.0
    
    if x <= min_x: return params[0]
    if x >= max_x: return params[-1]
    
    # map x to index
    normalized = (x - min_x) / (max_x - min_x) # 0 to 1
    idx = normalized * (n - 1)
    i = int(idx)
    t = idx - i
    
    # Bounds check
    if i >= n - 1:
        return params[-1]
    
    # Interpolate between params[i] and params[i+1]
    val = params[i] * (1 - t) + params[i+1] * t
    return val

class FeedForwardNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals # List of (node, activation_func, bias, [(input_node, weight), ...])
        self.values = {}

    def activate(self, inputs):
        if len(inputs) != len(self.input_nodes):
            raise ValueError("Expected matches number of inputs")
            
        self.values.clear()
        for i, node_id in enumerate(self.input_nodes):
            self.values[node_id] = inputs[i]

        for node_id, activation_func, bias, incoming in self.node_evals:
            total = bias
            for conn in incoming:
                # conn is (in_node, weight, gating_node, act_name, act_params)
                in_node = conn[0]
                weight = conn[1]
                gating_node = conn[2]
                act_name = conn[3] if len(conn) > 3 else 'identity'
                act_params = conn[4] if len(conn) > 4 else []
                
                val = self.values.get(in_node, 0.0)
                
                # KAN: Apply edge activation
                if act_name == 'bspline':
                    val = bspline_eval(val, act_params)
                elif act_name != 'identity':
                    f = activation_registry.get(act_name)
                    if f: val = f(val)

                if gating_node is not None:
                     gate_val = self.values.get(gating_node, 0.0)
                     val = val * gate_val
                     
                total += val * weight
            
            self.values[node_id] = activation_func(total)

        return [self.values[k] for k in self.output_nodes]

    def gate(self, gating_node_id, src_id, tgt_id):
        """
        Dynamically gates a connection.
        :param gating_node_id: The node that will modulate the connection.
        :param src_id: Source of the connection.
        :param tgt_id: Target of the connection.
        """
        # Find the connection in node_evals
        found = False
        for i, (node_id, activation_func, bias, incoming) in enumerate(self.node_evals):
            if node_id == tgt_id:
                for j, conn in enumerate(incoming):
                    # conn is tuple, need to replace it
                    if conn[0] == src_id:
                        # Reconstruct tuple with gating node, preserving KAN info
                        # len(conn) is likely 5 now
                        act_name = conn[3] if len(conn) > 3 else 'identity'
                        act_params = conn[4] if len(conn) > 4 else []
                        new_conn = (conn[0], conn[1], gating_node_id, act_name, act_params)
                        incoming[j] = new_conn
                        found = True
                        break
            if found: break
        
        if not found:
            # This might happen if connection doesn't exist.
            pass

    @staticmethod
    def create(genome):
        # Topological Sort to ensure feed-forward execution order
        input_nodes = [n.id for n in genome.nodes.values() if n.type == 'INPUT']
        output_nodes = [n.id for n in genome.nodes.values() if n.type == 'OUTPUT']
        
        # Build adjacency
        # Only use ENABLED connections
        connections = [c for c in genome.connections.values() if c.enabled]
        
        # Map node_id -> list of (incoming_node_id, weight)
        incoming_connections = {node_id: [] for node_id in genome.nodes}
        for conn in connections:
            if conn.out_node not in incoming_connections:
                 incoming_connections[conn.out_node] = []
            # Append (in_node, weight, gating_node, act, params)
            incoming_connections[conn.out_node].append((conn.in_node, conn.weight, conn.gating_node, conn.activation, conn.activation_params))
            
        # Kahn's Algorithm / DFS for Topo Sort
        # We need a list of nodes to evaluate in order.
        # Nodes that depend only on inputs come first.
        
        # Find all nodes reachable from inputs and that reach outputs (pruning)
        # But for simplicity, just sort all nodes.
        
        visited = set()
        eval_order = []
        
        # Helper for DFS
        def dfs(node_id, path_stack):
            if node_id in visited:
                return
            
            # Simple cycle checking if needed, but we rely on genome construction
            # being mostly Acyclic or ignoring back connections if we want pure FF.
            # But the 'creates_cycle' check in Genome ensures DAG.
            
            visited.add(node_id)
            
            # Visit dependencies first
            # Dependencies are nodes that feed INTO this node
            deps = incoming_connections.get(node_id, [])
            for conn in deps:
                dep_id = conn[0]
                dfs(dep_id, path_stack)
            
            eval_order.append(node_id)

        # Start DFS from outputs to ensure we only calculate what's needed for outputs
        for out_node in output_nodes:
            dfs(out_node, set())
            
        # Construct the efficient node_evals list
        node_evals = []
        for node_id in eval_order:
            node = genome.nodes[node_id]
            if node.type == 'INPUT':
                continue # Inputs are set manually
                
            activ_func = activation_registry.get(node.activation)
            incoming = incoming_connections.get(node_id, [])
            node_evals.append((node_id, activ_func, node.bias, incoming))
            
        return FeedForwardNetwork(input_nodes, output_nodes, node_evals)


class RecurrentNetwork:
    def __init__(self, inputs, outputs, node_keys, connections):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_keys = node_keys # All node IDs
        
        # Connection structure: node_id -> list of (in_node, weight)
        # For recurrent, we can't just use topo sort. We need to evaluate all nodes.
        self.connections = connections 
        
        # State: node_id -> current_value
        self.values = {k: 0.0 for k in node_keys}
        # Buffer for next time step
        self.next_values = {k: 0.0 for k in node_keys}

    def reset(self):
        for k in self.values:
            self.values[k] = 0.0
            self.next_values[k] = 0.0
            
    def activate(self, inputs):
        if len(inputs) != len(self.input_nodes):
            raise ValueError("Expected matches number of inputs")
            
        # Set inputs for current step
        # Inputs usually overwrite whatever was in the node value
        for i, node_id in enumerate(self.input_nodes):
            self.values[node_id] = inputs[i]
            
        # Calculate next values
        # We iterate over all nodes that are NOT inputs (inputs are clamped)
        # Wait, if an input node has incoming connection (recurrence?), standard NEAT inputs don't have incoming.
        # So we update HIDDEN and OUTPUT nodes.
        
        for node_id in self.node_keys:
            if node_id in self.input_nodes:
                 continue
                 
            # Retrieve node params (need to store them or pass them?)
            # Refactoring: RecurrentNetwork needs to know activation func and bias per node.
            # My current Init args are insufficient. 
            pass

    @staticmethod
    def create(genome):
        # All nodes
        node_keys = list(genome.nodes.keys())
        input_nodes = [n.id for n in genome.nodes.values() if n.type == 'INPUT']
        output_nodes = [n.id for n in genome.nodes.values() if n.type == 'OUTPUT']
        
        # Pre-process node params
        node_params = {} # id -> (activation, bias)
        from hyperneat.activations import registry as activation_registry
        
        for nid, node in genome.nodes.items():
            node_params[nid] = (activation_registry.get(node.activation), node.bias)
            
        # Connections
        incoming_conns = {nid: [] for nid in node_keys}
        for conn in genome.connections.values():
            if conn.enabled:
                # Append (in_node, weight, gating_node, act, params)
                incoming_conns[conn.out_node].append((conn.in_node, conn.weight, conn.gating_node, conn.activation, conn.activation_params))
                
        return RecurrentNetworkFull(input_nodes, output_nodes, node_keys, node_params, incoming_conns)

    @staticmethod
    def create_from_substrate(inputs, outputs, node_keys, node_params, incoming_conns):
        return RecurrentNetworkFull(inputs, outputs, node_keys, node_params, incoming_conns)

class RecurrentNetworkFull:
    def __init__(self, inputs, outputs, node_keys, node_params, incoming_conns):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_keys = node_keys
        self.node_params = node_params
        self.incoming_conns = incoming_conns
        
        self.values = {k: 0.0 for k in node_keys}
        
    def reset(self):
        for k in self.values:
            self.values[k] = 0.0
            
    def activate(self, inputs):
        if len(inputs) != len(self.input_nodes):
            raise ValueError("Expected matches number of inputs")
            
        # Load inputs
        for i, node_id in enumerate(self.input_nodes):
            self.values[node_id] = inputs[i]
            
        # Compute activations for next time step
        # Note: In synchronous update, we calculate all 'pre-activations' based on current 'values', 
        # then apply activation to get new values.
        
        next_values = {}
        
        for node_id in self.node_keys:
            if node_id in self.input_nodes:
                # Inputs preserve their value 
                next_values[node_id] = self.values[node_id]
                continue
                
            activ_func, bias = self.node_params[node_id]
            
            total = bias
            for conn in self.incoming_conns.get(node_id, []):
                src_id = conn[0]
                weight = conn[1]
                gating_node = conn[2] 
                act_name = conn[3] if len(conn) > 3 else 'identity'
                act_params = conn[4] if len(conn) > 4 else []
                
                val = self.values.get(src_id, 0.0)
                
                # Clamp input value to prevent overflow
                val = max(-1e6, min(1e6, val))
                
                # KAN Activation
                if act_name == 'bspline':
                    val = bspline_eval(val, act_params)
                elif act_name != 'identity':
                    f = activation_registry.get(act_name)
                    if f: val = f(val)
                
                if gating_node is not None:
                    gate_val = self.values.get(gating_node, 0.0)
                    val = val * gate_val
                    
                total += val * weight
            
            # Clamp total before activation
            total = max(-1e6, min(1e6, total))
            result = activ_func(total)
            
            # Handle NaN/inf in result
            import math
            if math.isnan(result) or math.isinf(result):
                result = 0.0
            
            next_values[node_id] = result
            
        self.values = next_values
        
        return [self.values[k] for k in self.output_nodes]

    def gate(self, gating_node_id, src_id, tgt_id):
        """
        Dynamically gates a connection.
        """
        if tgt_id in self.incoming_conns:
            incoming = self.incoming_conns[tgt_id]
            for i, conn in enumerate(incoming):
                if conn[0] == src_id:
                    act_name = conn[3] if len(conn) > 3 else 'identity'
                    act_params = conn[4] if len(conn) > 4 else []
                    new_conn = (conn[0], conn[1], gating_node_id, act_name, act_params)
                    incoming[i] = new_conn
                    return

