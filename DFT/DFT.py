# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
import cv2
import math
import cmath
class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        #print(matrix[0,0]);
        N = matrix.shape
        #print(N[0])
        outp = 0
        #medium = 0
        output = np.zeros((N[0],N[1]),dtype=np.complex)
        #print(output)
        for u in range(0,N[0]):
            for v in range(0,N[1]):
               outp = 0
               for k in range(0,N[0]):
                  for l in range(0,N[1]):
                    a = ((2 * math.pi)/N[0]) * ((u * k)+(v * l))
                    cosval = math.cos(a)
                    sinval = math.sin(a)
                    # print(1j*sinval)
                    #medium = medium + matrix[k,l] * (cosval - (1j*sinval))
                    outp = outp + (matrix[k,l] * (cosval - (1j*sinval)))
                    output[u,v] = outp
        #z = np.zeros((15,15),dtype= np.complex)
        #print'hello(np.allclose(output,np.fft.fft(matrix)))
        #print("hello",np.fft.fft2(matrix))

        return output

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        N = matrix.shape
        outp = 0
        # medium = 0
        output = np.zeros((N[0], N[1]), dtype=np.complex)
        # print(output)
        for u in range(0, N[0]):
            for v in range(0, N[1]):
                outp = 0
                for k in range(0, N[0]):
                    for l in range(0, N[1]):
                        a = ((2 * math.pi) / N[0]) * ((u * k) + (v * l))
                        cosval = math.cos(a)
                        sinval = math.sin(a)
                        # print(1j*sinval)
                        # medium = medium + matrix[k,l] * (cosval - (1j*sinval))
                        outp = outp + (matrix[k, l] * (cosval + (1j * sinval)))
                        output[u, v] = outp

        return output


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        N = matrix.shape
        outp = 0
        # medium = 0
        output = np.zeros((N[0], N[1]), dtype=np.uint8)
        # print(output)
        for u in range(0, N[0]):
            for v in range(0, N[1]):
                outp = 0
                for k in range(0, N[0]):
                    for l in range(0, N[1]):
                        a = ((2 * math.pi) / N[0]) * ((u * k) + (v * l))
                        cosval = math.cos(a)

                        # print(1j*sinval)
                        # medium = medium + matrix[k,l] * (cosval - (1j*sinval))
                        outp = outp + (matrix[k, l] * (cosval))
                        output[u, v] = outp


        return output


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        N = matrix.shape
        outp = 0
        magnitude = np.zeros((N[0], N[1]), np.uint8)
        # medium = 0
        output = np.zeros((N[0], N[1]), dtype=np.complex)
        # print(output)
        for u in range(0, N[0]):
            for v in range(0, N[1]):
                outp = 0
                for k in range(0, N[0]):
                    for l in range(0, N[1]):
                        a = ((2 * math.pi) / N[0]) * ((u * k) + (v * l))
                        cosval = math.cos(a)
                        sinval = math.sin(a)
                        # print(1j*sinval)
                        # medium = medium + matrix[k,l] * (cosval - (1j*sinval))
                        outp = outp + (matrix[k, l] * (cosval - (1j * sinval)))
                        output[u, v] = outp
        for p in range(0,N[0]):
            for q in range(0,N[1]):
                magnitude[p,q] = abs(output[p,q])

        #print(magnitude)


        return magnitude