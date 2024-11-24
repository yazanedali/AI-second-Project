import math
def Sigmoid(x):
    return 1/(1+math.exp(-x))
def leaniar(x):
    return x
def step(x):
    return 1 if x>=0 else 0
def sign(x):
    return 1 if x>=0 else -1
def tanh(x):
    return 1/(1+math.exp(-2*x))-1
def Rlu(x):
        return x if x>=0 else 0
