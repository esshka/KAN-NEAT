# Visualization module for HyperNEAT
# WHY: Allows users to visualize the geometry and connectivity of the substrate.
# RELEVANT FILES: hyperneat/substrate.py, hyperneat/phenotype.py

import matplotlib.pyplot as plt
from hyperneat.phenotype import FeedForwardNetwork, RecurrentNetworkFull

def draw_net(substrate, network=None, filename=None, show=True):
    """
    Draws the substrate nodes and optionally the network connections.
    
    :param substrate: The Substrate object.
    :param network: (Optional) The phenotype network (FeedForward or Recurrent).
    :param filename: (Optional) Filename to save the plot.
    :param show: (Optional) Whether to call plt.show().
    """
    
    plt.figure(figsize=(10, 10))
    plt.title("HyperNEAT Substrate")
    
    # 1. Draw Nodes
    
    # Inputs (Green)
    ix = [c[0] for c in substrate.input_coords]
    iy = [c[1] for c in substrate.input_coords]
    plt.scatter(ix, iy, c='green', s=100, label='Input', zorder=10)
    
    # Hidden (Blue)
    hx = []
    hy = []
    for layer in substrate.hidden_layers:
        for c in layer:
            hx.append(c[0])
            hy.append(c[1])
    if hx:
        plt.scatter(hx, hy, c='blue', s=80, label='Hidden', zorder=10)
        
    # Outputs (Red)
    ox = [c[0] for c in substrate.output_coords]
    oy = [c[1] for c in substrate.output_coords]
    plt.scatter(ox, oy, c='red', s=100, label='Output', zorder=10)
    
    # Add coordinate labels
    all_coords = substrate.input_coords + [c for layer in substrate.hidden_layers for c in layer] + substrate.output_coords
    all_ids = substrate.input_ids + substrate.all_hidden_and_output_ids
    
    # Create map for easy lookup
    id_to_coord = {i: c for i, c in zip(all_ids, all_coords)}
    
    for i in all_ids:
        x, y = id_to_coord[i]
        plt.text(x, y+0.05, str(i), fontsize=9, ha='center')

    # 2. Draw Connections
    if network:
        import matplotlib.patches as patches
        ax = plt.gca()
        
        connections = [] # (src_id, tgt_id, weight, gating_node_id)
        
        if isinstance(network, FeedForwardNetwork):
            for node_id, _, _, incoming in network.node_evals:
                for conn in incoming:
                    # conn is (src, weight) or (src, weight, gate)
                    src = conn[0]
                    w = conn[1]
                    gate = conn[2] if len(conn) > 2 else None
                    connections.append((src, node_id, w, gate))
                    
        elif isinstance(network, RecurrentNetworkFull):
            for tgt_id, incoming in network.incoming_conns.items():
                for conn in incoming:
                    src = conn[0]
                    w = conn[1]
                    gate = conn[2] if len(conn) > 2 else None
                    connections.append((src, tgt_id, w, gate))
                    
        max_weight = 3.0 # Assumed scaling
        
        for src, tgt, w, gate in connections:
            if src not in id_to_coord or tgt not in id_to_coord:
                continue
                
            p1 = id_to_coord[src]
            p2 = id_to_coord[tgt]
            
            color = 'red' if w < 0 else 'blue'
            alpha = min(1.0, abs(w) / max_weight) + 0.1
            if alpha > 1.0: alpha = 1.0
            width = max(0.1, min(3.0, abs(w)))
            
            # Determine curvature style
            if src == tgt:
                # Self loop
                # Offset start and end to create a visible loop
                # Shift p1 left, p2 right
                p1_mod = (p1[0] - 0.05, p1[1])
                p2_mod = (p1[0] + 0.05, p1[1])
                # Large radius to loop "up" or "around"
                connectionstyle = "arc3,rad=0.5"
                
                arrow = patches.FancyArrowPatch(
                    p1_mod, p2_mod,
                    connectionstyle=connectionstyle,
                    color=color,
                    alpha=alpha,
                    linewidth=width,
                    arrowstyle='->',
                    mutation_scale=15,
                    linestyle='dashed',
                    zorder=1
                )
            else:
                # Standard connection
                connectionstyle = "arc3,rad=0.1"
                
                arrow = patches.FancyArrowPatch(
                    p1, p2,
                    connectionstyle=connectionstyle,
                    color=color,
                    alpha=alpha,
                    linewidth=width,
                    arrowstyle='->',
                    mutation_scale=15,
                    linestyle='dashed',
                    zorder=1
                )
            
            ax.add_patch(arrow)
            
            # Draw Gating Line if applicable
            if gate is not None and gate in id_to_coord:
                gate_pos = id_to_coord[gate]
                # Midpoint of connection (approximate)
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                
                # Draw yellow/orange line from Gate Node to Connection Midpoint
                plt.plot([gate_pos[0], mid_x], [gate_pos[1], mid_y], 
                         color='orange', linestyle=':', linewidth=1.5, alpha=0.8, zorder=5)
                # Scatter point at midpoint
                plt.scatter([mid_x], [mid_y], c='orange', s=30, zorder=6, marker='s')

    # Create proxy artists for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Hidden', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Pos Weight (Gated)'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Neg Weight (Gated)'),
        Line2D([0], [0], marker=r'$\circlearrowright$', color='w', label='Recurrent', markerfacecolor='black', markeredgecolor='black', markersize=15),
        Line2D([0], [0], color='orange', linestyle=':', lw=2, label='Dynamic Gate'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Set padding
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    pad = 0.5
    plt.xlim(min(xs) - pad, max(xs) + pad)
    plt.ylim(min(ys) - pad, max(ys) + pad)
    
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        
    if show:
        plt.show()
    else:
        plt.close()
