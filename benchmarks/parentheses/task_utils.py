# Shared Utils for Nested Parentheses Task
# WHY: Ensures consistent problem definition across benchmarks.
# RELEVANT FILES: None

import random

# ==============================================================================
# TASK DEFINITION
# ==============================================================================
# Input: Sequence of tokens
#  0: '('
#  1: ')'
# Output:
#  0: Invalid
#  1: Valid (Balanced)

INPUT_SIZE = 2
OUTPUT_SIZE = 2 # 0=Invalid, 1=Valid (using softmax/classification)

class ParenthesesTask:
    def __init__(self, max_depth=4, sequence_length=10):
        self.max_depth = max_depth
        self.sequence_length = sequence_length
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        
    def generate_balanced(self):
        """Generates a balanced string."""
        if self.max_depth == 0: return ""
        
        # Simple recursive generation
        # S -> (S)S | epsilon
        
        # To strictly control length is tricky with pure recursion.
        # Let's generate a valid stack sequence.
        
        seq = []
        balance = 0
        length = 0
        target_len = random.randint(2, self.sequence_length)
        if target_len % 2 != 0: target_len += 1 # Must be even
        
        while length < target_len:
            if balance == 0:
                seq.append(0) # Must add '('
                balance += 1
            elif balance > 0:
                # Can close or open (if depth allows)
                # Bias towards closing if getting long
                prob_close = 0.5
                if length > target_len - balance - 2: prob_close = 1.0
                if balance >= self.max_depth: prob_close = 1.0
                
                if random.random() < prob_close:
                    seq.append(1) # ')'
                    balance -= 1
                else:
                    seq.append(0) # '('
                    balance += 1
            length += 1
            
        # Force close pending
        while balance > 0:
            seq.append(1)
            balance -= 1
            
        return seq

    def generate_unbalanced(self):
        """Generates an unbalanced string."""
        # Mix of random chars
        length = random.randint(2, self.sequence_length)
        seq = [random.choice([0, 1]) for _ in range(length)]
        
        # Check if accidentally valid
        if self.is_valid(seq):
            # Break it
            if len(seq) > 0:
                seq[0] = 1 if seq[0] == 0 else 0
        return seq

    def is_valid(self, seq):
        balance = 0
        for token in seq:
            if token == 0: balance += 1
            else:
                balance -= 1
                if balance < 0: return False
        return balance == 0
        
    def generate_example(self):
        """Returns (inputs, target_label)"""
        # 50/50 Valid/Invalid
        if random.random() < 0.5:
            seq = self.generate_balanced()
            label = 1 # Valid
        else:
            seq = self.generate_unbalanced()
            label = 0 # Invalid
            
        # Convert inputs to one-hot
        inputs = []
        for token in seq:
            vec = [0.0] * INPUT_SIZE
            vec[token] = 1.0
            inputs.append(vec)
            
        return inputs, label

    def evaluate_fitness(self, net, num_examples=20):
        total_error = 0.0
        hits = 0.0
        
        for _ in range(num_examples):
            inputs, target_label = self.generate_example()
            net.reset()
            
            # Feed sequence
            output = None
            for inp in inputs:
                output = net.activate(inp)
            
            # Use state at FINAL step
            if output is None: # Empty seq
                output = [0.0] * OUTPUT_SIZE
                
            # Check NaNs
            import math
            if any(math.isnan(x) or math.isinf(x) for x in output):
                return 0.001
            
            # Target Vector
            target_vec = [0.0] * OUTPUT_SIZE
            target_vec[target_label] = 1.0
            
            # Loss (MSE)
            mse = sum((t - o)**2 for t, o in zip(target_vec, output))
            total_error += mse
            
            # Accuracy
            pred = output.index(max(output))
            if pred == target_label:
                hits += 1
                
        # Fitness: Accuracy primary, error secondary
        accuracy = hits / num_examples
        fit = accuracy * 10.0
        if accuracy > 0.9:
            avg_mse = total_error / num_examples
            fit += (1.0 - avg_mse) * 5.0
            
        return max(0.001, fit)

def verify_visual(net):
    print("\nVerifying Parentheses Task")
    task = ParenthesesTask()
    
    for _ in range(5):
        inputs, target = task.generate_example()
        net.reset()
        
        seq_str = ""
        for vec in inputs:
            char = '(' if vec[0]==1.0 else ')'
            seq_str += char
            net.activate(vec)
            
        # Final output only
        output = net.activate(inputs[-1]) # Just re-run last step? 
        # Wait, recurrent nets state is persistent.
        # We activate sequentially. The last call to activate() returned the final output.
        # BUT standard pattern is:
        # for x in inputs: out = net.activate(x)
        # So 'output' variable holds result of last step.
        
        # Correction: `task.evaluate_fitness` correctly captures last output.
        # Here we just need to capture it.
        
        # Net state is preserved? Yes.
        # But to show output we need it.
        # Let's re-run cleanly.
        net.reset()
        out = None
        for vec in inputs:
            out = net.activate(vec)
            
        pred = out.index(max(out))
        label_str = "Valid" if target == 1 else "Invalid"
        pred_str = "Valid" if pred == 1 else "Invalid"
        check = "✓" if pred == target else "✗"
        
        print(f"Seq: {seq_str:20} | Tgt: {label_str:7} | Pred: {pred_str:7} ({max(out):.2f}) {check}")
