import numpy as np
from scipy import optimize
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

def gaussian(amp, x_mean, y_mean, x_width, y_width, offset):
    x_width = float(x_width)
    y_width = float(y_width)

    return lambda x, y: amp*np.exp(-0.5*((x-x_mean)/x_width)**2-0.5*((y-y_mean)/y_width)**2) + offset

# return a 2D gaussian fit
# generally a 2D gaussian fit can have 7 params, 6 of them are implemented here (the excluded one is an angle)
# codes adapted from https://scipy-cookbook.readthedocs.io/items/FittingData.html
def gaussianfit(data, roi, showimg=False):
    # calculate moments for initial guess
    data = data[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
    if showimg:
        plt.imshow(data)
        plt.show()

    total = np.sum(data)
    X, Y = np.indices(data.shape)
    x_mean = np.sum(X*data)/total
    y_mean = np.sum(Y*data)/total
    col = data[:, int(y_mean)]
    x_width = np.sqrt(np.abs((np.arange(col.size)-x_mean)**2*col).sum()/col.sum())
    row = data[int(x_mean), :]
    y_width = np.sqrt(np.abs((np.arange(row.size)-y_mean)**2*row).sum()/row.sum())
    offset = (data[0, :].sum()+data[-1, :].sum()+data[:, 0].sum()+data[:, -1].sum())/np.sum(data.shape)/2
    amp = data.max() - offset

    # use optimize function to obtain 2D gaussian fit
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape))-data)
    p, success = optimize.leastsq(errorfunction, (amp, x_mean, y_mean, x_width, y_width, offset))

    p_dict = {}
    p_dict["x_mean"] = p[1]
    p_dict["y_mean"] = p[2]
    p_dict["x_width"] = p[3]
    p_dict["y_width"] = p[4]
    p_dict["amp"] = p[0]
    p_dict["offset"] = p[5]

    return p_dict

class atomnumanalysis:
    def __init__(self, fname, gname, detuning):
        # resonant_cross_section in mm^2, linewidth in MHz
        param = {"pixeltomm": 0.099, "kB": 1.38064852e-23, "m": 86.9*1.66053873e-27, "confidence_band": 0.95, "resonant_cross_section": 1.356e-7, "linewidth":6.065}

        atom_num = self.readhdf(fname, gname, param, detuning)

        # atom_num = np.random.rand(10)+3

        mpl.style.use("seaborn")
        self.fig, self.ax = plt.subplots()
        self.plot(atom_num, param=param)
        self.ax.set_xlabel("No. of images")
        self.ax.set_ylabel("Atom number")
        self.ax.set_title(gname)
        self.ax.legend()
        # plt.hist(atom_num, bins=40)
        plt.show()

    def readhdf(self, fname, gname, param, detuning):
        with h5py.File(fname, "r") as f:
            group = f[gname]
            atom_num = np.array([])
            cross_section = param["resonant_cross_section"]/(1+4*(detuning/param["linewidth"])**2)

            for img in group.keys():
                img_data = group[img]
                roi = {"xmin":105, "xmax":175, "ymin":75, "ymax":155} # choose a braod roi for the first fit trial

                fitresult = gaussianfit(img_data, roi)
                new_roi = {} # calculate a new roi based on the first fit result (use +/-3sigma region)
                new_roi["xmin"] = int(np.maximum(roi["xmin"]+fitresult["x_mean"]-3*fitresult["x_width"], 0))
                new_roi["xmax"] = int(np.minimum(roi["xmin"]+fitresult["x_mean"]+3*fitresult["x_width"], img_data.shape[0]))
                new_roi["ymin"] = int(np.maximum(roi["ymin"]+fitresult["y_mean"]-3*fitresult["y_width"], 0))
                new_roi["ymax"] = int(np.minimum(roi["ymin"]+fitresult["y_mean"]+3*fitresult["y_width"], img_data.shape[1]))

                fitresult = gaussianfit(img_data, new_roi, showimg=False) # make a second fit using the new roi
                roi = new_roi
                new_roi = {} # calculate a new roi based on the second fit result (use +/-3sigma region)
                new_roi["xmin"] = int(np.maximum(roi["xmin"]+fitresult["x_mean"]-3*fitresult["x_width"], 0))
                new_roi["xmax"] = int(np.minimum(roi["xmin"]+fitresult["x_mean"]+3*fitresult["x_width"], img_data.shape[0]))
                new_roi["ymin"] = int(np.maximum(roi["ymin"]+fitresult["y_mean"]-3*fitresult["y_width"], 0))
                new_roi["ymax"] = int(np.minimum(roi["ymin"]+fitresult["y_mean"]+3*fitresult["y_width"], img_data.shape[1]))

                sc = np.sum(img_data[new_roi["xmin"]:new_roi["xmax"], new_roi["ymin"]:new_roi["ymax"]]) # signal count
                atom_num = np.append(atom_num, sc*(param["pixeltomm"]**2)/cross_section)

        return atom_num

    def plot(self, atom_num, param={}):
        color = 'C1'
        self.ax.plot(atom_num, marker='o', mfc=color, markeredgewidth=0.8, markeredgecolor='k', linestyle='')

        c = stats.norm.ppf((1+param["confidence_band"])/2) # 95% confidence level gives critical value c=1.96
        mean = np.mean(atom_num)
        std = np.std(atom_num)/np.sqrt(len(atom_num))
        x = np.arange(len(atom_num))
        self.ax.plot(x, np.ones(len(atom_num))*mean, color, label="Atom number: "+np.format_float_scientific(mean, precision=2)+"("+ np.format_float_scientific(std, precision=1)+")")
        self.ax.fill_between(x, np.ones(len(atom_num))*(mean-c*std), np.ones(len(atom_num))*(mean+c*std), color=color, alpha=0.2, label="{:.0f}% confidence band".format(param["confidence_band"]*100))


filepath = "C:/Users/dur!p5/github/pixelfly-python-control/saved_images/"
filename = "images_20210922.hdf"
fname = filepath + filename
gname = "rfabsrp_lambda_20210922_190640"
detuning = 26 # in MHz

# calculate and plot temperature, inital rms radius, reduced \chi^2, 1-CDF(\chi^2)
# indicate uncertainties at "confidence_band" confidence level
# plot pointwise confident band at "confidence_band" level
tof = atomnumanalysis(fname, gname, detuning)
