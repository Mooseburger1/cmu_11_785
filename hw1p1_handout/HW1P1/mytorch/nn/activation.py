import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z.copy()

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):
        self.A = Z.copy()
        
        return 1 / (1 + np.exp(-self.A))
    
    def backward(self):
        return self.forward(self.A) * (1 - self.forward(self.A))


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """

    def forward(self, Z):
        self.A = Z.copy()

        return (np.exp(self.A) - np.exp(-self.A)) / (np.exp(self.A) + np.exp(-self.A))
    
    def backward(self):
        return 1 - self.forward(self.A)**2


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        self.A = Z.copy()

        relu = self.A
        relu[relu <= 0] = 0
        return relu

        
    
    def backward(self):
        relu = np.ones_like(self.A)
        return relu * (self.A > 0).astype(float)
