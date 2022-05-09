import numpy as np
from scipy import optimize
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from matplotlib import cm


class imageave:
    def __init__(self):
        param = {"pixeltomm": 0.107, "kB": 1.38064852e-23, "m": 86.9*1.66053873e-27, "confidence_band": 0.95}
        roi = {"xmin":90, "xmax":190, "ymin":70, "ymax":170}

        filepath = "C:/Users/dur!p5/github/pixelfly-python-control/saved_images/"
        cmotimages = {"fname": "images_20210922.hdf", "gname":"rffluor_cmot_20210922_173145", "subgname":"instr no. 8_1000000.0"}

        image_ave_cmot = self.readhdf(filepath, cmotimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.indices(image_ave_cmot.shape)
        im_cmot = ax.plot_surface(X, Y, image_ave_cmot, cmap=cm.coolwarm)
        # plt.colorbar(im_cmot)
        plt.show()

    def readhdf(self, filepath, hdfinfo):
        with h5py.File(filepath+hdfinfo["fname"], "r") as f:
            image_list = f[hdfinfo["gname"]]
            for i, img in enumerate(image_list.keys()):
                if i == 0:
                    image_total = image_list[img][:, :]
                else:
                    image_total += image_list[img][:, :]
            image_ave = image_total/len(image_list)

        return image_ave

tof = imageave()
