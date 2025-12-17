# Shared Utils for Inverse Kinematics (2-Link Arm) Task
# WHY: Ensures consistent physics and evaluation across benchmarks.
# RELEVANT FILES: None

import math
import random

# ==============================================================================
# TASK DEFINITION
# ==============================================================================
# 2-Link Arm
# L1 = 0.5, L2 = 0.5 (Total reach = 1.0)
# Joint Limits: 0 to Pi (0 to 180 degrees)

# INPUTS: Target X, Target Y (in reach)
# OUTPUTS: Theta1, Theta2 (normalized 0-1)

INPUT_SIZE = 2
OUTPUT_SIZE = 2

class InverseKinematicsTask:
    def __init__(self):
        self.L1 = 0.5
        self.L2 = 0.5
        
    def forward_kinematics(self, t1, t2):
        # t1, t2 in radians
        x = self.L1 * math.cos(t1) + self.L2 * math.cos(t1 + t2)
        y = self.L1 * math.sin(t1) + self.L2 * math.sin(t1 + t2)
        return x, y

    def generate_example(self):
        # Generate valid target by picking random angles
        # This ensures the target is always reachable
        t1_tgt = random.uniform(0, math.pi)
        t2_tgt = random.uniform(0, math.pi)
        
        target_x, target_y = self.forward_kinematics(t1_tgt, t2_tgt)
        
        return [target_x, target_y], (t1_tgt, t2_tgt)

    def evaluate_fitness(self, net, num_examples=50):
        total_error = 0.0
        
        # Consistent seed for evaluation if possible? No, evaluating generalization.
        
        for _ in range(num_examples):
            inputs, (t1_true, t2_true) = self.generate_example()
            
            # Activate Net
            # Provide inputs
            # For FF nets, activate once. For Recurrent, activate(inputs)
            # Assuming create_phenotype returns a net with activate(inputs) -> outputs
            
            # Note: RecurrentNetwork.activate takes inputs, updates state, returns output
            # FeedForwardNetwork.activate takes inputs, returns output
            
            # net.reset() # Important for recurrent nets, harmless for FF
            if hasattr(net, 'reset'):
                net.reset()
            output = net.activate(inputs)
            
            # Check mechanics
            if isinstance(output, list) and len(output) != OUTPUT_SIZE:
                 # Some implementations might return dict or full state?
                 # Assuming standard list return
                 pass
            
            # Protection against NaN
            if any(math.isnan(x) or math.isinf(x) for x in output):
                return 0.001

            # Decode Outputs (Sigmoid 0-1 -> 0-Pi)
            # Clip to be safe
            p1 = max(0.0, min(1.0, output[0]))
            p2 = max(0.0, min(1.0, output[1]))
            
            theta1 = p1 * math.pi
            theta2 = p2 * math.pi
            
            # Forward Kinematics to see where we ended up
            actual_x, actual_y = self.forward_kinematics(theta1, theta2)
            
            # Distance Error
            dist = math.sqrt((inputs[0] - actual_x)**2 + (inputs[1] - actual_y)**2)
            total_error += dist
            
        avg_error = total_error / num_examples
        # Fitness: Maximize (1 - Error)
        # Perfect score = 100.0 (error=0)
        # Allow negative fitness? Usually NEAT likes positive.
        
        fitness = 1.0 / (0.01 + avg_error) 
        # If error is 0.0, fitness is 100.
        # If error is 1.0 (max reach), fitness is ~1.
        
        return fitness

def verify_visual(net):
    print("\nVerifying Inverse Kinematics")
    task = InverseKinematicsTask()
    
    total_dist = 0
    for _ in range(5):
        inputs, (t1_t, t2_t) = task.generate_example()
        if hasattr(net, 'reset'):
            net.reset()
        output = net.activate(inputs)
        
        p1 = max(0.0, min(1.0, output[0]))
        p2 = max(0.0, min(1.0, output[1]))
        t1, t2 = p1 * math.pi, p2 * math.pi
        
        ax, ay = task.forward_kinematics(t1, t2)
        tx, ty = inputs
        
        dist = math.sqrt((tx-ax)**2 + (ty-ay)**2)
        total_dist += dist
        
        print(f"Tgt: ({tx:.2f}, {ty:.2f}) | Act: ({ax:.2f}, {ay:.2f}) | Err: {dist:.4f}")
    
    print(f"Avg Error: {total_dist/5:.4f}")
