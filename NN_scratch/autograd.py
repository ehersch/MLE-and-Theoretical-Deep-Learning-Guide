import numpy as np
import math

class Value:
    """
        Defines object of Value class.
        Takes in data (float), children (set), label and _op (strings)
        Each node takes on a gradient and _backprop function for future backprop implementation.
    """
    def __init__(self, data:float, _children=(), label='', _op=''):
        self.data = data
        self._prev = set(_children)
        self.label = label
        self._op = _op
        
        ## backprop and grad as well
        self.grad = 0
        self._backprop = lambda: None
        
    def __repr__(self):
        """
            Represent each Value object as label and data
        """
        return f'{self.label}:{self.data}'
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data + other.data, (self, other), _op = '+')
        
        def backprop():
            self.grad += 1 * result.grad
            other.grad += 1 * result.grad
        
        result._backprop = backprop
        return result
    
    def __radd__(self, other):
        # Accomodate for any ordering of addition
        return self + other
    
    def __neg__(self):
        result = Value(-1 * self.data, (self,), label=f'-{self.label}', _op = '-')
        
        def backprop():
            self.grad += -1 * result.grad
            
        result._backprop = backprop
        return result
    
    def __sub__(self, other):
        result = self + (-other)
        result._op = '-'
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        result = Value(self.data * other.data, (self, other), _op = '*')
        
        def backprop():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        
        result._backprop = backprop
        
        return result
    
    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        result = self * (other**(-1))
        result._op = '/'
        return result
    
    def __pow__(self, power:float):
        result = Value(self.data ** power, (self,), _op = f'^{power}')
        
        def backprop():
            self.grad += (power) * self.data ** (power - 1) * result.grad
            
        result._backprop = backprop
        return result
    
    def exp(self):
        result = Value(math.exp(self.data), (self,), _op = f'exp')
        
        def backprop():
            self.grad += result.data * result.grad
            
        result._backprop = backprop
        return result
    
    def tanh(self):
        # tanh = (e^x - e^(-x)) / (e^x + e^(-x))... OR
        result = Value(math.tanh(self.data), (self,), _op = f'tanh')
        
        def backprop():
            self.grad += (1 - result.data**2) * result.grad
        
        result._backprop = backrop
        return result
    
    def relu(self):
        # max(0, x)
        result = Value(max(0, self.data), (self,), _op = 'relu')
        
        def backprop():
            if self.data > 0:
                self.grad += result.grad
                
        result._backprop = backprop   
        return result        
        
def topo_sort(root):
    """
        Topological sort of nodes in computation graph, ending with root.
    """
    visited = set([])
    ordering = []
    
    def dfs(node):
        if node not in visited:
            visited.add(node)
            
            for child in node._prev:
                dfs(child)
                
            ordering.append(node)
    dfs(root)
    return ordering

def backward(root):
    """
        Given root, iterate through topo-graph (backwards) calling backprop.
        Alters the gradients at each node in computation graph.
    """
    ordering = topo_sort(root)
    
    for cur in ordering:
        cur.grad = 0
        
    root.grad = 1
    
    for cur in reversed(ordering):
        if cur._prev is not None:
            cur._backprop()