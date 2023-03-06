import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros(shape=(out_features, in_features)) # M x N
        self.b = np.zeros(shape=(out_features, 1)) # M x 1

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # (batch, N)
        self.N = A.shape[0] # (batch)
        self.Ones = np.ones((self.N,1)) # (batch, 1)
        Z = (self.A @ self.W.T )+ (self.Ones @ self.b.T) # ((batch, N) x (N x M)) + ((batch, 1) x (1, M)) => (batch,M) + (batch, M) => (batch, M)


        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T # (N x M)
        dZdW = self.A
        dZdb = np.ones((self.N, 1))  # (out_features, 1)

        dLdA = dLdZ @ dZdA.T  # (batch, M) x (M, N)
        dLdW = dLdZ.T @ dZdW  # (M, batch) x (batch, N)
        dLdb = dLdZ.T @ dZdb  # (M, batch) x (batch, 1)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
