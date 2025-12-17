# Activation functions for Neural Networks
# WHY: Provides non-linearities for the NEAT/CPPN nodes.
# RELEVANT FILES: hyperneat/genome.py, hyperneat/phenotype.py

import math
import random

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-4.9 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def tanh(x):
    # Math.tanh is safe in recent python versions usually, but manual clamp for safety
    try:
        return math.tanh(x)
    except OverflowError:
        return -1.0 if x < 0 else 1.0

def sin(x):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return math.sin(x)

def gaussian(x):
    try:
        # Clamp x^2 to avoid overflow
        x_sq = min(x * x, 50.0)
        return math.exp(-x_sq / 2.0)
    except OverflowError:
        return 0.0

def identity(x):
    return x

def relu(x):
    return x if x > 0 else 0

def abs_val(x):
    return abs(x)

class ActivationRegistry:
    def __init__(self):
        self.functions = {}
        self.register("sigmoid", sigmoid)
        self.register("tanh", tanh)
        self.register("sin", sin)
        self.register("gaussian", gaussian)
        self.register("identity", identity)
        self.register("relu", relu)
        self.register("abs", abs_val)


    def register(self, name, func):
        self.functions[name] = func

    def get(self, name):
        return self.functions.get(name, sigmoid)

    def get_random_name(self):
        return random.choice(list(self.functions.keys()))

# Additional Novel Functions
def cos_act(x):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return math.cos(x)

def step(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def swish(x):
    return x * sigmoid(x)

def softplus(x):
    try:
        if x > 20: return x # Avoid overflow for large x
        return math.log(1 + math.exp(x))
    except OverflowError:
        return x

def clamped(x):
    return max(-1.0, min(1.0, x))

def inv(x):
    try:
        return 1.0 / x
    except ZeroDivisionError:
        return 0.0

def log_act(x):
    try:
        val = abs(x)
        if val < 1e-7: val = 1e-7
        return math.log(val)
    except ValueError:
        return 0.0

def exp_act(x):
    try:
        # Clamp input to avoid overflow
        if x > 20: return 1e9 
        return math.exp(x)
    except OverflowError:
        return 1e9 if x > 0 else 0.0

def square(x):
    return x * x

def cube(x):
    return x * x * x

def hat(x):
    return max(0.0, 1.0 - abs(x))

def bspline_dummy(x):
    return x # Real implementation in phenotype with params

# Global registry instance
registry = ActivationRegistry()
registry.register("cos", cos_act)
registry.register("step", step)
registry.register("leaky_relu", leaky_relu)
registry.register("swish", swish)
registry.register("softplus", softplus)
registry.register("clamped", clamped)
registry.register("inv", inv)
registry.register("log", log_act)
registry.register("exp", exp_act)
registry.register("square", square)
registry.register("cube", cube)
registry.register("hat", hat)
registry.register("bspline", bspline_dummy)
