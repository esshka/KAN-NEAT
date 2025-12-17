# HyperNEAT Substrate
# WHY: Defines the geometry of the problem and maps CPPN outputs to network weights.
# RELEVANT FILES: hyperneat/phenotype.py


from hyperneat.phenotype import FeedForwardNetwork, RecurrentNetwork
from hyperneat.activations import registry as activation_registry

class Substrate:
    def __init__(self, input_coords, output_coords, hidden_coords_layers=None, allow_recurrence=False, use_leo=False, gating_node_id=None, gating_mode="global"):
        """
        :param input_coords: List of (x, y) tuples for input nodes.
        :param output_coords: List of (x, y) tuples for output nodes.
        :param hidden_coords_layers: List of Lists of (x, y) tuples. Each list is a hidden layer.
        :param allow_recurrence: If True, allows connections between all nodes in the substrate (except inputs to inputs).
        :param use_leo: If True, uses Link Expression Output (gating). CPPN must output 2 values: [weight, bias/gate].
        :param gating_node_id: ID of the node that will be assigned as 'gate' if CPPN requests it (used if gating_mode="global").
        :param gating_mode: "global" (single fixed gate) or "geometric" (nearest node to midpoint).
        """
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.hidden_layers = hidden_coords_layers if hidden_coords_layers else []
        self.allow_recurrence = allow_recurrence
        self.use_leo = use_leo
        self.gating_node_id = gating_node_id
        self.gating_mode = gating_mode
        
        # Assign IDs to all nodes in the target substrate network
        self.input_ids = list(range(len(input_coords)))
        
        current_id = len(input_coords)
        self.hidden_ids = []
        for layer in self.hidden_layers:
            ids = list(range(current_id, current_id + len(layer)))
            self.hidden_ids.append(ids)
            current_id += len(layer)
            
        self.output_ids = list(range(current_id, current_id + len(output_coords)))
        
        # Flatten all IDs for easier recurrent queries
        self.all_hidden_and_output_ids = [i for layer in self.hidden_ids for i in layer] + self.output_ids
        self.all_ids = self.input_ids + self.all_hidden_and_output_ids

        # Map ID to Coordinate for geometric calculations
        all_coords = self.input_coords + [c for layer in self.hidden_layers for c in layer] + self.output_coords
        self.id_to_coord = {i: all_coords[i] for i in range(len(all_coords))}
        
        # Pre-calculate all connections to query
        self.connections_to_query = []
        
        if self.allow_recurrence:
            # Query ALL pairs (src, tgt) where tgt is not an Input node.
            for src_id in self.all_ids:
                for tgt_id in self.all_hidden_and_output_ids:
                     self.connections_to_query.append({
                         'input': self.id_to_coord[src_id] + self.id_to_coord[tgt_id],
                         'src_id': src_id,
                         'tgt_id': tgt_id
                     })
                     
        else:
            # Standard Feed-Forward Layer-wise
            layers = [self.input_coords] + self.hidden_layers + [self.output_coords]
            layer_ids = [self.input_ids] + self.hidden_ids + [self.output_ids]
            
            for i in range(len(layers) - 1):
                source_layer = layers[i]
                target_layer = layers[i+1]
                source_ids = layer_ids[i]
                target_ids = layer_ids[i+1]
                
                for src_idx, src_coord in enumerate(source_layer):
                    for tgt_idx, tgt_coord in enumerate(target_layer):
                         # (x1, y1, x2, y2, src_id, tgt_id)
                         self.connections_to_query.append({
                             'input': src_coord + tgt_coord, 
                             'src_id': source_ids[src_idx],
                             'tgt_id': target_ids[tgt_idx]
                         })

    def get_geometric_gate(self, src_pos, tgt_pos):
        """Find the node closest to the midpoint of the connection."""
        midpoint = ((src_pos[0] + tgt_pos[0]) / 2, (src_pos[1] + tgt_pos[1]) / 2)
        
        best_node = None
        min_dist = float('inf')
        
        # Search all hidden nodes as candidates for gates
        candidates = [i for layer in self.hidden_ids for i in layer]
        if not candidates:
             candidates = self.all_hidden_and_output_ids # Fallback
             
        for node_id in candidates:
             pos = self.id_to_coord[node_id]
             dist = (pos[0] - midpoint[0])**2 + (pos[1] - midpoint[1])**2
             if dist < min_dist:
                 min_dist = dist
                 best_node = node_id
                 
        return best_node

    def build_network(self, cppn):
        """
        Builds a Standard Network based on weights from the CPPN.
        """
        threshold = 0.2 
        weight_scale = 6.0 
        leo_threshold = 0.0 
        
        # Store adjacency: tgt_id -> list of (src_id, weight, gate_id, act_name, params)
        adjacency = {node_id: [] for node_id in self.all_ids}
        
        for conn in self.connections_to_query:
            outputs = cppn.activate(conn['input'])
            
            created = False
            final_weight = 0.0
            
            is_self_loop = (conn['src_id'] == conn['tgt_id'])
            eff_threshold = 0.05 if is_self_loop else threshold
            
            if self.use_leo:
                 if len(outputs) < 2:
                     w = outputs[0]
                     if abs(w) > eff_threshold:
                         final_weight = w * weight_scale
                         created = True
                 else:
                     w = outputs[0]
                     leo_val = outputs[1]
                     
                     if leo_val > leo_threshold or (is_self_loop and leo_val > -0.5):
                         if abs(w) > eff_threshold:
                             final_weight = w * weight_scale
                             created = True
            else:
                w = outputs[0]
                if abs(w) > eff_threshold:
                    final_weight = w * weight_scale
                    created = True
            
            # Check for Gating
            gate_id = None
            if 'gate_id' in conn:
                gate_id = conn['gate_id']
            else:
                gating_idx = -1
                if self.use_leo:
                     if len(outputs) >= 3: gating_idx = 2
                else:
                     if len(outputs) >= 2: gating_idx = 1
                
                should_gate = False
                if gating_idx != -1 and gating_idx < len(outputs):
                     gating_val = outputs[gating_idx]
                     if gating_val > 0.0:
                         should_gate = True
                
                if should_gate:
                    if self.gating_mode == "global" and self.gating_node_id is not None:
                        gate_id = self.gating_node_id
                    elif self.gating_mode == "geometric":
                        src_pos = self.id_to_coord[conn['src_id']]
                        tgt_pos = self.id_to_coord[conn['tgt_id']]
                        gate_id = self.get_geometric_gate(src_pos, tgt_pos)
            
            if 'weight' in conn:
                final_weight = conn['weight']
                created = True 
                
            if created:
                # Standard Substrate always uses Identity edges
                adjacency[conn['tgt_id']].append((conn['src_id'], final_weight, gate_id, 'identity', []))

        return self._create_phenotype(cppn, adjacency)

    def _create_phenotype(self, cppn, adjacency):
        # Helper to query node params (Bias, Activation)
        node_params = {} 
        bias_scale = 6.0 
        
        nodes_to_query = self.all_hidden_and_output_ids
        func_names = sorted(list(activation_registry.functions.keys()))
        n_funcs = len(func_names)

        for node_id in nodes_to_query:
             coord = self.id_to_coord[node_id]
             bias_input = [coord[0], coord[1], 0.0, 0.0]
             bias_output = cppn.activate(bias_input)
             
             bias = 0.0
             bias_idx = -1
             if len(bias_output) >= 4:
                 bias_idx = 3 
             elif len(bias_output) == 3:
                 bias_idx = 2 
                 
             if bias_idx != -1 and bias_idx < len(bias_output):
                 bias = bias_output[bias_idx] * bias_scale
                 
             activ_func = activation_registry.get('sigmoid')
             act_idx_in_output = 4 
             
             if len(bias_output) > act_idx_in_output:
                 act_val = bias_output[act_idx_in_output]
                 type_idx = int(abs(act_val * n_funcs)) % n_funcs
                 activ_func = activation_registry.get(func_names[type_idx])
             
             node_params[node_id] = (activ_func, bias)

        if self.allow_recurrence:
             return RecurrentNetwork.create_from_substrate(
                 self.input_ids, self.output_ids, self.all_ids, node_params, adjacency
             )
        else:
            node_evals = [] 
            for layer_ids in self.hidden_ids:
                for node_id in layer_ids:
                    activ_func, bias = node_params.get(node_id, (activation_registry.get('sigmoid'), 0.0))
                    incoming = adjacency[node_id]
                    node_evals.append((node_id, activ_func, bias, incoming))
                    
            for node_id in self.output_ids:
                activ_func, bias = node_params.get(node_id, (activation_registry.get('sigmoid'), 0.0))
                incoming = adjacency[node_id]
                node_evals.append((node_id, activ_func, bias, incoming))
                
            return FeedForwardNetwork(self.input_ids, self.output_ids, node_evals)

    @staticmethod
    def create_from_genome(genome, input_coords, output_coords, substrate_class=None):
        """
        Creates a Substrate based on the genome's substrate_config.
        """
        if substrate_class is None:
            substrate_class = Substrate

        hidden_coords_layers = []
        
        config = getattr(genome, 'substrate_config', {})
        hidden_layers = config.get('hidden_layers', [])
        
        num_layers = len(hidden_layers) + 2
        
        for i, layer_size in enumerate(hidden_layers):
            layer_idx = i + 1 # 0 is input
            x = -1.0 + (2.0 * layer_idx / (num_layers - 1))
            
            layer_coords = []
            if layer_size == 1:
                layer_coords.append((x, 0.0))
            else:
                for j in range(layer_size):
                    y = -1.0 + (2.0 * j / (layer_size - 1))
                    layer_coords.append((x, y))
            
            hidden_coords_layers.append(layer_coords)
            
        return substrate_class(input_coords, output_coords, hidden_coords_layers, allow_recurrence=True)


class KANSubstrate(Substrate):
    def __init__(self, input_coords, output_coords, hidden_coords_layers=None, allow_recurrence=False, use_leo=False, gating_node_id=None, gating_mode="global"):
        super().__init__(input_coords, output_coords, hidden_coords_layers, allow_recurrence, use_leo, gating_node_id, gating_mode)

    def build_network(self, cppn):
        """
        Builds a KAN Network where edges can have B-Spline activations.
        """
        threshold = 0.2 
        weight_scale = 6.0 
        leo_threshold = 0.0 
        
        adjacency = {node_id: [] for node_id in self.all_ids}
        
        for conn in self.connections_to_query:
            outputs = cppn.activate(conn['input'])
            
            created = False
            final_weight = 0.0
            
            is_self_loop = (conn['src_id'] == conn['tgt_id'])
            eff_threshold = 0.05 if is_self_loop else threshold
            
            if self.use_leo:
                 if len(outputs) < 2:
                     w = outputs[0]
                     if abs(w) > eff_threshold:
                         final_weight = w * weight_scale
                         created = True
                 else:
                     w = outputs[0]
                     leo_val = outputs[1]
                     if leo_val > leo_threshold or (is_self_loop and leo_val > -0.5):
                         if abs(w) > eff_threshold:
                             final_weight = w * weight_scale
                             created = True
            else:
                w = outputs[0]
                if abs(w) > eff_threshold:
                    final_weight = w * weight_scale
                    created = True
            
            # Gating
            gate_id = None
            if 'gate_id' in conn:
                gate_id = conn['gate_id']
            else:
                gating_idx = -1
                if self.use_leo:
                     if len(outputs) >= 3: gating_idx = 2
                else:
                     if len(outputs) >= 2: gating_idx = 1
                
                should_gate = False
                if gating_idx != -1 and gating_idx < len(outputs):
                     gating_val = outputs[gating_idx]
                     if gating_val > 0.0: should_gate = True
                
                if should_gate:
                    if self.gating_mode == "global" and self.gating_node_id is not None:
                        gate_id = self.gating_node_id
                    elif self.gating_mode == "geometric":
                        src_pos = self.id_to_coord[conn['src_id']]
                        tgt_pos = self.id_to_coord[conn['tgt_id']]
                        gate_id = self.get_geometric_gate(src_pos, tgt_pos)
            
            if 'weight' in conn:
                final_weight = conn['weight']
                created = True 
                
            if created:
                # --- Hyper-KAN-NEAT: Edge Activation ---
                edge_act_name = 'identity'
                edge_act_params = []
                
                edge_act_idx = 5
                if edge_act_idx < len(outputs):
                    edge_act_val = outputs[edge_act_idx]
                    abs_val = abs(edge_act_val)
                    
                    if abs_val < 0.6:
                        edge_act_name = 'identity'
                    elif abs_val < 0.9:
                        norm_sub = (abs_val - 0.6) / 0.3 
                        common_types = ['sigmoid', 'tanh', 'relu']
                        type_idx = int(norm_sub * len(common_types)) % len(common_types)
                        edge_act_name = common_types[type_idx]
                    else:
                        edge_act_name = 'bspline'
                    
                    if edge_act_name == 'bspline':
                        for pi in range(6, 10):
                            if pi < len(outputs):
                                edge_act_params.append(outputs[pi])
                            else:
                                edge_act_params.append(0.0)
                
                adjacency[conn['tgt_id']].append((conn['src_id'], final_weight, gate_id, edge_act_name, edge_act_params))

        return self._create_phenotype(cppn, adjacency)


