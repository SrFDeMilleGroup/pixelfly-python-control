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

def linear(x, slope, offset):
    return slope*x+offset

def linearfit(x, y, yerr):
    ind = np.argsort(x)
    x_sorted = x[ind]
    y_sorted = y[ind]
    slope = (y_sorted[-1]-y_sorted[0])/(x_sorted[-1]-x_sorted[0])
    offset = (y_sorted[0]*x_sorted[-1]-y_sorted[-1]*x_sorted[0])/(x_sorted[-1]-x_sorted[0])

    popt, pcov = optimize.curve_fit(linear, x, y, p0=(slope, offset), sigma=yerr)

    return (popt, pcov)

class gravityanalysis:
    def __init__(self, fname, gname):
        param = {"g": 9.8e-3, "confidence_band": 0.95} # g is gravitational acceleration in mm/ms^2

        time_sq, axial_position, axial_position_err, radial_position, radial_position_err = self.readhdf(fname, gname, param)

        # time_sq = time_sq[2:]
        # axial_position = axial_position[2:]
        # axial_position_err = axial_position_err[2:]
        # radial_position = radial_position[2:]
        # radial_position_err = radial_position_err[2:]

        mpl.style.use("seaborn")
        self.fig, self.ax = plt.subplots()
        self.plot(time_sq, radial_position, radial_position_err, type="radial", param=param)
        self.plot(time_sq, axial_position, axial_position_err, type="axial", param=param)
        self.ax.set_xlabel("time of flight$^2$ [ms$^2$]")
        self.ax.set_ylabel("posiiton [pixel]")
        self.ax.set_title(gname)
        self.ax.legend()
        plt.show()

    def readhdf(self, fname, gname, param):
        with h5py.File(fname, "r") as f:
            group = f[gname]
            time_sq = np.array([])
            axial_position = np.array([])
            axial_position_err = np.array([])
            radial_position = np.array([])
            radial_position_err = np.array([])

            for subg in group.keys():
                image_list = group[subg]
                x_position = np.array([])
                y_position = np.array([])
                for img in image_list.keys():
                    img_data = group[subg][img]
                    roi = {"xmin":20, "xmax":260, "ymin":20, "ymax":200} # choose a braod roi for the first fit trial
                    fitresult = gaussianfit(img_data, roi)
                    new_roi = {} # calculate a new roi based on the first fit result (use +/-3sigma region)
                    new_roi["xmin"] = int(np.maximum(roi["xmin"]+fitresult["x_mean"]-3*fitresult["x_width"], 0))
                    new_roi["xmax"] = int(np.minimum(roi["xmin"]+fitresult["x_mean"]+3*fitresult["x_width"], img_data.shape[0]))
                    new_roi["ymin"] = int(np.maximum(roi["ymin"]+fitresult["y_mean"]-3*fitresult["y_width"], 0))
                    new_roi["ymax"] = int(np.minimum(roi["ymin"]+fitresult["y_mean"]+3*fitresult["y_width"], img_data.shape[1]))
                    fitresult = gaussianfit(img_data, new_roi, showimg=False) # make a second fit using the new roi
                    x_position =np.append(x_position, new_roi["xmin"]+fitresult["x_mean"])
                    y_position =np.append(y_position, new_roi["ymin"]+fitresult["y_mean"])
                axial_position = np.append(axial_position, np.mean(y_position))
                axial_position_err = np.append(axial_position_err, np.std(y_position)/np.sqrt(len(y_position)))
                radial_position = np.append(radial_position, np.mean(x_position))
                radial_position_err = np.append(radial_position_err, np.std(x_position)/np.sqrt(len(x_position)))
                time_sq = np.append(time_sq, (float(subg.split("_")[-1])/1e6)**2) # convert ns to ms

        return (time_sq, axial_position, axial_position_err, radial_position, radial_position_err)

    def plot(self, time_sq, position, position_err, type="", param={}):
        if type == "radial":
            color = 'C1'
        elif type == "axial":
            color = 'C2'
        else:
            print("Plot type not supported.")
            return

        popt, pcov = linearfit(time_sq, position, position_err)
        fit_chisq = np.sum(((linear(time_sq, *popt)-position)/position_err)**2)
        reduced_chisq = fit_chisq/(len(time_sq)-2)
        # gof = 100*(1 - stats.chi2.cdf(fit_chisq, len(time_sq)-2)) # in percent, goodness of fit, see https://faculty1.coloradocollege.edu/~sburns/toolbox/DataFitting.html
        self.ax.errorbar(time_sq, position, yerr=position_err, marker='o', mfc=color, markeredgewidth=0.8, markeredgecolor='k', ecolor=color, linestyle='')

        x = np.linspace(0, np.amax(time_sq), 200)
        position_fit = linear(x, *popt)
        c = stats.norm.ppf((1+param["confidence_band"])/2) # 95% confidence level gives critical value c=1.96
        perr = np.sqrt(np.diag(pcov)) # gives the standard deviation of fitting parameters
        pixeltomm = -param["g"]/2/popt[0] # convert to uK
        pixeltomm_err = param["g"]/2/(popt[0]**2)*perr[0]
        label = type + ": {:.4f}({:.4f}) mm, $\chi^2_\\nu$: {:.2f}".format(pixeltomm, pixeltomm_err, reduced_chisq)
        self.ax.plot(x, position_fit, color, label=label)

        k, b = unc.correlated_values(popt, pcov)
        py = k*x+b
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        self.ax.fill_between(x, nom-c*std, nom+c*std, color=color, alpha=0.2, label="{:.0f}% confidence band".format(param["confidence_band"]*100))


filepath = "C:/Users/dur!p5/github/pixelfly-python-control/saved_images/"
filename = "images_20210610.hdf"
fname = filepath + filename
gname = "molasses_20210610_145852"

# calculate and plot temperature, inital rms radius, reduced \chi^2, 1-CDF(\chi^2)
# indicate uncertainties at "confidence_band" confidence level
# plot pointwise confident band at "confidence_band" level
tof = gravityanalysis(fname, gname)
