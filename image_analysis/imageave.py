import numpy as np
from scipy import optimize
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py


class imageave:
    def __init__(self):
        param = {"pixeltomm": 0.107, "kB": 1.38064852e-23, "m": 86.9*1.66053873e-27, "confidence_band": 0.95}
        roi = {"xmin":90, "xmax":190, "ymin":70, "ymax":170}

        filepath = "C:/Users/dur!p5/github/pixelfly-python-control/saved_images/"
        motimages = {"fname": "images_20210818.hdf", "gname":"mot_tof_20210818_201732", "subgname":"instr no. 7_1000000.0"}
        cmotimages = {"fname": "images_20210818.hdf", "gname":"cmot_tof_20210818_202648", "subgname":"instr no. 8_1000000.0"}
        bmolimages = {"fname": "images_20210818.hdf", "gname":"broght_molasses_tof_20210818_203735", "subgname":"instr no. 9_5000000.0"}
        lambdaimages = {"fname": "images_20210818.hdf", "gname":"lambda_tof_20210818_205322", "subgname":"instr no. 10_7000000.0"}

        image_ave_mot = self.readhdf(filepath, motimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
        image_ave_cmot = self.readhdf(filepath, cmotimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
        image_ave_bmol = self.readhdf(filepath, bmolimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
        image_ave_lambda = self.readhdf(filepath, lambdaimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]

        fig, [ax_mot, ax_cmot, ax_bmol, ax_lambda] = plt.subplots(1, 4)
        im_mot = ax_mot.imshow(image_ave_mot)
        im_cmot = ax_cmot.imshow(image_ave_cmot)
        im_bmol = ax_bmol.imshow(image_ave_bmol)
        im_lambda = ax_lambda.imshow(image_ave_lambda)
        fig.colorbar(im_mot)
        plt.show()

    def readhdf(self, filepath, hdfinfo):
        with h5py.File(filepath+hdfinfo["fname"], "r") as f:
            image_list = f[hdfinfo["gname"]][hdfinfo["subgname"]]
            for i, img in enumerate(image_list.keys()):
                if i == 0:
                    image_total = image_list[img][:, :]
                else:
                    image_total += image_list[img][:, :]
            image_ave = image_total/len(image_list)

        return image_ave

tof = imageave()
