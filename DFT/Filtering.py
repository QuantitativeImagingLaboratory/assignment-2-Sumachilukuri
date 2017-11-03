# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import cv2
import numpy as np
import math
import cmath
import sys
import matplotlib.pyplot as plt
class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        c = shape[1]
        r = shape[0]
        mask = np.zeros((r, c), np.uint8)
        for u in range(r):
            for v in range(c):
                value = ((u - (r/2)) ** 2 + (v - (c/2)) ** 2) ** (1 / 2)
                if (value <= cutoff):
                    mask[u, v] = 1
                else:
                    mask[u, v] = 0

        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        il_mask = self.get_ideal_low_pass_filter(shape, cutoff)
        mask = 1 - il_mask

        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        c = shape[1]
        r = shape[0]
        n = 2*self.order
        #print(n)
        mask = np.zeros((r, c), np.uint8)
        for u in range(r):
            for v in range(c):
                value = ((u - (r / 2)) ** 2 + (v - (c / 2)) ** 2) ** (1 / 2)
                mask[u, v] = 1/(1+((value/cutoff)**(n)))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        #bl_mask = self.get_ideal_low_pass_filter(shape, cutoff)
        #mask = 1 - bl_mask
        c = shape[1]
        r = shape[0]
        n = 2 * self.order
        # print(n)
        mask = np.zeros((r, c), dtype=float)
        for u in range(r):
            for v in range(c):
                value = ((u - (r / 2)) ** 2 + (v - (c / 2)) ** 2) ** (1 / 2)
                try:
                 mask[u, v] = 1.0 / (1 + ((cutoff / value) ** (n)))
                except ZeroDivisionError:
                 mask[u,v] = 0


        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        c = shape[1]
        r = shape[0]
        mask = np.zeros((r, c))
        for u in range(r):
            for v in range(c):
                value = ((u - (r / 2)) ** 2 + (v - (c / 2)) ** 2) ** (1 / 2)
                mask[u, v] = 1 / (math.exp(value ** 2 / (2 * (cutoff ** 2))))

        #print("mask",mask)
        #for u in range(r):
            #for v in range(c):
                #if(mask[u,v]!=0):
                    #print(u)
                    #print(v)
                    #print("hello")


        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        gl_mask = self.get_gaussian_low_pass_filter(shape, cutoff)
        mask = 1 - gl_mask
        
        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """

        return image


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        input = self.image
        k = input.shape
        img_fft = np.fft.fft2(input)
        #print(img_fft)
        self.shift = np.fft.fftshift(img_fft)
        #print(shift)
        magnitude = np.zeros((k[0], k[1]),dtype = np.float)
        dft_log = np.zeros((k[0], k[1]),dtype = np.uint8)
        magnitude = abs(self.shift)
        #print(magnitude)
        dft_log = np.log(1+magnitude)
        coeff = (255) / (dft_log.max() - (dft_log.min()))
        coeff1 = dft_log - (dft_log.min())
        #print(coeff1)
        cont_stret = coeff * coeff1
        #print(cont_stret)
        #plt.imshow(cont_stret,cmap='gray')
        #plt.show()
        if (self.filter == self.get_ideal_low_pass_filter):
            mask = self.get_ideal_low_pass_filter(self.shift.shape, self.cutoff)
        elif (self.filter == self.get_ideal_high_pass_filter):
            mask = self.get_ideal_high_pass_filter(self.shift.shape, self.cutoff)
        elif (self.filter == self.get_butterworth_low_pass_filter):
            mask = self.get_butterworth_low_pass_filter(self.shift.shape, self.cutoff, self.order)
        elif (self.filter == self.get_butterworth_high_pass_filter):
            mask = self.get_butterworth_high_pass_filter(self.shift.shape, self.cutoff, self.order)
        elif (self.filter == self.get_gaussian_low_pass_filter):
            mask = self.get_gaussian_low_pass_filter(self.shift.shape, self.cutoff)
        elif (self.filter == self.get_gaussian_high_pass_filter):
            mask = self.get_gaussian_high_pass_filter(self.shift.shape, self.cutoff)
        else:
            print("Give a valid filter")
        mask_shift = mask*self.shift

        msize = mask_shift.shape
        mask_abs = np.zeros((msize[0], msize[1]), np.uint8)
        mask_abs = np.log(1 + abs(mask_shift))
        #mask_abs = abs(mask_shift)
        maskcoeff = (255) / (mask_abs.max() - mask_abs.min())
        maskcoeff1 = mask_abs - (mask_abs.min())
        mask_strech = maskcoeff * maskcoeff1
        #plt.imshow(mask_strech, cmap="gray")
        #plt.show()
        #mask_inverse = np.zeros((msize[0], msize[1]), dtype=np.complex)
        mask_inverse = np.fft.ifft2(np.fft.ifftshift(mask_shift))

        mask_invabs = np.zeros((msize[0], msize[1]), dtype=np.uint8)
        #mask_invabs = np.log(1 + abs(mask_inverse))
        mask_invabs = abs(mask_inverse)
        #print(mask_invabs)
        #mask_invstrech1 = np.zeros((msize[0], msize[1]), dtype=np.uint8)
        mask_invstrech = np.zeros((msize[0], msize[1]), dtype=np.float)
        maskinv_coeff = (255) / (mask_invabs.max() - mask_invabs.min())
        maskinv_coeff1 = mask_invabs - (mask_invabs.min())
        mask_invstrech = maskinv_coeff * maskinv_coeff1
        #mask_invstrech1 = mask_invstrech
        #print(mask_invstrech)
        #plt.imshow(mask_invstrech, cmap="gray")
        #plt.show()

        return [cont_stret, mask_strech, mask_invstrech]
