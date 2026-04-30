from autograd import Value
import numpy as np

class Neuron:
    def __init__(self, weights:list, bias:Value=0):
        self.weights = weights
        self.bias = bias
        self.n = len(weights)
        
    def forward(self, x:list):
        ## w^Tx + b
        output = 0
        for i in range(self.n):
            output += self.weights[i] * x[i]
            
        output += self.bias
        return output
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, dim_in:int, dim_out:int):
        self.dim_out = dim_out
        self.dim_in = dim_in
        
        self.neurons = []
        
        ## layer represented as a list of neurons
        ## weights are N(0,1)
        for i in range(dim_out):
            self.neurons += [Neuron([Value(np.random.normal(0, 1) * 0.01) for _ in range(dim_in)], Value(0.0))]
                        
    def forward(self, x:list, nonlinearity=None):
        output = []
        
        ## w^Tx + b
        ## now imagine x is still a vector, but W is a matrix and b a vector
        ## W has dimension [dim_in, dim_out]
        
        for i in range(self.dim_out):
            neuron = self.neurons[i]
            output += [neuron.forward(x)]
            
        # print("Before relu:", output)
            
        if nonlinearity:
            for i in range(len(output)):
                output[i] = output[i].relu()
            
        return output
    

    def parameters(self):
        output = []
        
        for i in range(self.dim_out):
            output += [self.neurons[i].parameters()]
            
        return output
            
        
class MLP:
    def __init__(self, dim_in, layer_sizes:list, dim_out, non_linearities:list):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.non_linearities = non_linearities
        
        total_dimensions = [dim_in] + layer_sizes + [dim_out]
        
        for i in range(len(total_dimensions) - 1):
            self.layers += [Layer(total_dimensions[i], total_dimensions[i+1])]
                            
    def forward(self, x:list):
        cur_x = x
        for i in range(len(self.layer_sizes)):
            cur_x = self.layers[i].forward(cur_x, self.non_linearities[i])
            
        output = self.layers[-1].forward(cur_x)
                            
        return output
    
    def parameters(self):
        output = []
        for layer in self.layers:
            output += [layer.parameters()]
            
        return output