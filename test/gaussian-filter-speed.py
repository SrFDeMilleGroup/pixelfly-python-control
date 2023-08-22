from scipy.ndimage import correlate, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import time

class gaussian_filter_speed_test:
    def __init__(self):
        rng = np.random.default_rng(seed=0)
        arr = rng.random((348, 260))

        shape_x = 45
        shape_y = 45
        sigma = 7.5

        repeat = 20

        t1, arr_filtered_1 = self.method_1(arr, shape_x, shape_y, sigma, repeat)
        print("Method 1 takes time {:.2f} s.".format(t1))

        t2, arr_filtered_2 = self.method_2(arr, shape_x, shape_y, sigma, repeat)
        print("Method 2 takes time {:.2f} s.".format(t2))

        t3, arr_filtered_3 = self.method_3(arr, sigma, repeat)
        print("Method 3 takes time {:.2f} s.".format(t3))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14,5))

        ax1.imshow(arr)
        ax1.set_title("Unfiltered")

        ax2.imshow(arr_filtered_1)
        ax2.set_title("Filtered by method 1")

        ax3.imshow(arr_filtered_2)
        ax3.set_title("Filtered by method 2")

        ax4.imshow(arr_filtered_3)
        ax4.set_title("Filtered by method 3")

        plt.show()

    def method_1(self, arr, shape_x, shape_y, sigma, repeat=100):
        t0 = time.time()
        for _ in range(repeat):
            arr_filter = correlate(arr, self.matlab_style_gauss2D((shape_x, shape_y), sigma))

        return time.time()-t0, arr_filter

    def method_2(self, arr, shape_x, shape_y, sigma, repeat=100):
        h = self.matlab_style_gauss2D((shape_x, shape_y), sigma)
        t0 = time.time()
        for _ in range(repeat):
            arr_filter = correlate(arr, h)

        return time.time()-t0, arr_filter

    def method_3(self, arr, sigma, repeat=100):
        t0 = time.time()
        for _ in range(repeat):
            arr_filter = gaussian_filter(arr, sigma)

        return time.time()-t0, arr_filter

    def matlab_style_gauss2D(self, shape=(3,3),sigma=0.5):
        """
        https://stackoverflow.com/a/17201686

        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


gf = gaussian_filter_speed_test()
