# Shared Utils for Copy Task
# WHY: Ensures all benchmark variants solve the exact same Copy Task problem.
# RELEVANT FILES: None

import random

# ==============================================================================
# TASK DEFINITION
# ==============================================================================

# Symbols
# 0: Token A
# 1: Token B
# 2: Delimiter
INPUT_SIZE = 3
OUTPUT_SIZE = 2  # Predict A or B (Delimiter is input-only trigger)

class CopyTask:
    def __init__(self, sequence_length=3, delay_length=3):
        self.sequence_length = sequence_length
        self.delay_length = delay_length
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
    
    def generate_example(self):
        """
        Generates a copy task example.
        Input:  [Seq] + [Delay] + [Delimiter]
        Target: [Zeros] + [Zeros] + [Seq]
        
        Returns:
            inputs: List of one-hot vectors
            targets: List of one-hot vectors
        """
        # 1. Sequence to recall (A or B)
        seq_tokens = [random.choice([0, 1]) for _ in range(self.sequence_length)]
        
        # 2. Build Input Sequence
        inputs = []
        
        # Phase 1: Input Sequence
        for token in seq_tokens:
            vec = [0.0] * INPUT_SIZE
            vec[token] = 1.0
            inputs.append(vec)
            
        # Phase 2: Delay (Inputs are zero)
        for _ in range(self.delay_length):
            inputs.append([0.0] * INPUT_SIZE)
            
        # Phase 3: Delimiter (Trigger recall)
        # Note: We present delimiter repeatedly for the length of sequence output
        # Or just once? Standard copy task usually presents delimiter once, then runs for N steps.
        # But for NEAT/RNN easy evaluation, we can present delimiter for N steps.
        # Let's use: Single Delimiter token, then N steps of Zeros (or maintain delimiter).
        # Let's stick to standard: Delimiter once, then network must output.
        # BUT for simpler NEAT evolution: Let's present Delimiter for N steps during recall.
        
        # REVISED Phase 3: Recall
        # We will feed "Delimiter" token for as many steps as we want output.
        for _ in range(self.sequence_length):
            vec = [0.0] * INPUT_SIZE
            vec[2] = 1.0  # Delimiter bit
            inputs.append(vec)
            
        # 3. Build Target Sequence
        # During Input + Delay: Target is 0 (or don't care)
        # We only evaluate during Recall phase.
        targets = []
        
        # Ignore initial steps
        for _ in range(self.sequence_length + self.delay_length):
            targets.append(None) # Don't care
            
        # Recall targets
        for token in seq_tokens:
            vec = [0.0] * OUTPUT_SIZE
            vec[token] = 1.0
            targets.append(vec)
            
        return inputs, targets

    def evaluate_fitness(self, net, num_examples=5, detailed=False):
        """
        Evaluates a network on the copy task.
        """
        total_error = 0.0
        total_hits = 0.0
        total_steps = 0
        
        valid = True
        
        for _ in range(num_examples):
            inputs, targets = self.generate_example()
            net.reset()
            
            # Run network
            outputs_history = []
            
            for i, inp in enumerate(inputs):
                output = net.activate(inp)
                outputs_history.append(output)
                
                # Check NaNs
                import math
                if any(math.isnan(x) or math.isinf(x) for x in output):
                    valid = False
                    break
                
                target = targets[i]
                if target is not None:
                    # Calculate error on recall steps
                    mse = sum((t - o)**2 for t, o in zip(target, output))
                    total_error += mse
                    total_steps += 1
                    
                    # Accuracy
                    pred_idx = output.index(max(output))
                    targ_idx = target.index(max(target))
                    if pred_idx == targ_idx:
                        total_hits += 1
            
            if not valid:
                break
                
        if not valid:
            return 0.0
        
        # Normalize
        avg_mse = total_error / max(1, total_steps)
        accuracy = total_hits / max(1, total_steps)
        
        if detailed:
            return avg_mse, accuracy, total_hits, total_steps
            
        # Fitness Function
        # Maximize accuracy, Minimize MSE
        # Base fitness = Accuracy * 10
        # Bonus = (1.0 - MSE) * 5 (if accuracy > 0.5)
        
        fitness = accuracy * 10
        if accuracy > 0.9: # Bonus for precision
            fitness += (1.0 - min(1.0, avg_mse)) * 5
            
        return max(0.001, fitness)

# Helper to verify output visually
def verify_visual(net, sequence_length=3, delay_length=5):
    task = CopyTask(sequence_length, delay_length)
    inputs, targets = task.generate_example()
    net.reset()
    
    print(f"\nVerifying Copy Task (Seq={sequence_length}, Delay={delay_length})")
    
    # Reconstruct input seq from inputs for display
    # First N are seq, Next M are delay (0), Last N are delimiter
    input_seq_tokens = []
    for i in range(sequence_length):
        vec = inputs[i]
        token = 0 if vec[0] == 1.0 else 1
        charmap = {0: 'A', 1: 'B'}
        input_seq_tokens.append(charmap[token])
        
    print(f"Sequence to Copy: {input_seq_tokens}")
    
    for i, inp in enumerate(inputs):
        out = net.activate(inp)
        tgt = targets[i]
        
        step_type = "INPUT"
        if i >= sequence_length and i < sequence_length + delay_length:
            step_type = "DELAY"
        elif i >= sequence_length + delay_length:
            step_type = "RECALL"
            
        tgt_str = " - "
        check = ""
        if tgt is not None:
            tgt_idx = tgt.index(max(tgt))
            tgt_str = "A" if tgt_idx == 0 else "B"
            
            out_idx = out.index(max(out))
            out_str = "A" if out_idx == 0 else "B"
            
            check = "✓" if out_idx == tgt_idx else "✗"
            print(f"{step_type:6} | In: {inp} | Out: {[f'{x:.2f}' for x in out]} ({out_str}) | Tgt: {tgt_str} {check}")
        # else:
            # print(f"{step_type:6} | In: {inp} | Out: {[f'{x:.2f}' for x in out]}")
