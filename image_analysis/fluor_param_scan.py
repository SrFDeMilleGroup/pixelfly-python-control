import numpy as np
from scipy import optimize
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
def gaussianfit(data, roi, showimg=False, normalize=False):
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

    if normalize:
        p_dict["x_width"] = p_dict["x_width"]/((total)**(1/3))
        p_dict["y_width"] = p_dict["y_width"]/((total)**(1/3))

    return p_dict

filepath = "C:/Users/dur!p5/github/pixelfly-python-control/saved_images/"
filename = "images_20210608.hdf"
groupname = "molassestof_20210608_193024"
pixeltomm = 0.107

with h5py.File(filepath+filename, "r") as f:
    group = f[groupname]
    scanparam = np.array([])
    axialwidth = np.array([])
    axialwidth_err = np.array([])
    radialwidth = np.array([])
    radialwidth_err = np.array([])
    scanparam_name = list(group.keys())[0].split("_")[0]

    for subg in group.keys():
        image_list = group[subg]
        x_width_list = np.array([])
        y_width_list = np.array([])
        for img in image_list.keys():
            img_data = group[subg][img]
            roi = {"xmin":20, "xmax":260, "ymin":20, "ymax":200}
            fitresult = gaussianfit(img_data, roi)
            new_roi = {}
            new_roi["xmin"] = int(np.maximum(roi["xmin"]+fitresult["x_mean"]-3*fitresult["x_width"], 0))
            new_roi["xmax"] = int(np.minimum(roi["xmin"]+fitresult["x_mean"]+3*fitresult["x_width"], img_data.shape[0]))
            new_roi["ymin"] = int(np.maximum(roi["ymin"]+fitresult["y_mean"]-3*fitresult["y_width"], 0))
            new_roi["ymax"] = int(np.minimum(roi["ymin"]+fitresult["y_mean"]+3*fitresult["y_width"], img_data.shape[1]))
            fitresult = gaussianfit(img_data, new_roi, showimg=False, normalize=False)
            # print(fitresult)
            x_width_list =np.append(x_width_list, fitresult["x_width"]*pixeltomm)
            y_width_list =np.append(y_width_list, fitresult["y_width"]*pixeltomm)

        sorted_index_array = np.argsort(y_width_list)
        # y_width_list = y_width_list[sorted_index_array][2:-2]
        # print(y_width_list)
        axialwidth = np.append(axialwidth, np.mean(y_width_list))
        axialwidth_err = np.append(axialwidth_err, np.std(y_width_list)/np.sqrt(len(y_width_list)))

        sorted_index_array = np.argsort(x_width_list)
        # x_width_list = x_width_list[sorted_index_array][2:-2]
        radialwidth = np.append(radialwidth, np.mean(x_width_list))
        radialwidth_err = np.append(radialwidth_err, np.std(x_width_list)/np.sqrt(len(x_width_list)))

        scanparam = np.append(scanparam, float(subg.split("_")[-1]))

mpl.style.use("seaborn")
plt.errorbar(scanparam, radialwidth, yerr=radialwidth_err, marker='o', mfc='C1', markeredgewidth=0.8, markeredgecolor='k', ecolor='C1', linestyle='', label="radial rms radius")
plt.errorbar(scanparam, axialwidth, yerr=axialwidth_err, marker='o', mfc='C2', markeredgewidth=0.8, markeredgecolor='k', ecolor='C2', linestyle='', label="axial rms radius")

plt.xlabel(scanparam_name)
plt.ylabel("cloud rms radius [mm]")
plt.legend()
plt.show()
