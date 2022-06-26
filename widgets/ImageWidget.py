import logging, traceback
import PyQt5
import pyqtgraph as pg
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
from matplotlib import cm
import numpy as np

# subclass the pyqtgraph rectangular ROI class so we can disable/modify some of its properties
class newRectROI(pg.RectROI):
    # see https://pyqtgraph.readthedocs.io/en/latest/graphicsItems/roi.html#pyqtgraph.ROI.checkPointMove
    def __init__(self, pos, size, centered=False, sideScalers=False, **args):
        super().__init__(pos, size, centered=False, sideScalers=False, **args)

    def setBounds(self, pos, size):
        bounds = PyQt5.QtCore.QRectF(pos[0], pos[1], size[0], size[1])
        self.maxBounds = bounds

    def setEnabled(self, arg):
        if not isinstance(arg, bool):
            logging.warning("Argument given in wrong type.")
            logging.warning(traceback.format_exc())
            return

        self.resizable = arg # set if ROI can be scaled
        self.translatable = arg # set if ROi can be translated

    def checkPointMove(self, handle, pos, modifiers):
        return self.resizable

class imageWidget:
    def __init__(self, parent, name, include_ROI=False, colorname="viridis", dummy_data_xmax=400, dummy_data_ymax=300):
        self.colormap = self.get_matplotlib_colormap(colorname=colorname, lut=6)

        self.graphlayout = pg.GraphicsLayoutWidget(parent=parent, border=True)
        plot = self.graphlayout.addPlot(row=1, col=1, rowspan=2, colspan=1, title=name)
        self.img = pg.ImageItem(lockAspect=True)
        plot.addItem(self.img)

        # add histogram/colorbar
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        hist.gradient.restoreState({'mode': 'rgb', 'ticks': self.colormap})

        # add a pyqt widget to graphlayout
        # https://stackoverflow.com/questions/45184941/how-do-i-add-a-qpushbutton-to-pyqtgraph-using-additem
        proxy = qt.QGraphicsProxyWidget()
        self.chb = qt.QCheckBox("Auto scale color")
        proxy.setWidget(self.chb)
        self.graphlayout.addItem(hist, row=1, col=2)
        self.graphlayout.addItem(proxy, row=2, col=2)
        # self.chb.setEnabled(False)

        self.dummy_data = self.generate_dummy_data(dummy_data_xmax, dummy_data_ymax)
        self.img.setImage(self.dummy_data)

        if include_ROI:
            # place in-image ROI selection
            self.img_roi = newRectROI(pos = [0, 0], 
                                size = [100, 100],
                                snapSize = 0,
                                scaleSnap = False,
                                translateSnap = False,
                                pen = "r")
                                # params ([x_start, y_start], [x_span, y_span])

            # add ROI scale handlers
            self.img_roi.addScaleHandle([0, 0], [1, 1])
            self.img_roi.addScaleHandle([1, 0], [0, 1])
            self.img_roi.addScaleHandle([0, 1], [1, 0])
            # params ([x, y], [x position scaled around, y position scaled around]), rectangular from 0 to 1
            self.img_roi.addScaleHandle([0, 0.5], [1, 0.5])
            self.img_roi.addScaleHandle([1, 0.5], [0, 0.5])
            self.img_roi.addScaleHandle([0.5, 0], [0.5, 1])
            self.img_roi.addScaleHandle([0.5, 1], [0.5, 0])
            plot.addItem(self.img_roi)

    def get_matplotlib_colormap(self, colorname="viridis", lut=6):
        color = cm.get_cmap(colorname, lut)
        colordata = color(range(lut)) # (r, g, b, a=opacity)
        colordata_reform = []
        for i in range(lut):
            l = [i/(lut-1), tuple([int(x*255) for x in colordata[i]])]
            colordata_reform.append(tuple(l))

        return colordata_reform

    def generate_dummy_data(self, xmax, ymax):
        # generate a fake image of 2D gaussian distribution image
        x_range=20
        y_range=20
        x_center=12
        y_center=16
        x_err=5
        y_err=4
        amp=100
        noise_amp = 10
        x, y = np.meshgrid(np.arange(x_range), np.arange(y_range))
        dst = np.sqrt((x-x_center)**2/(2*x_err**2)+(y-y_center)**2/2/(2*y_err**2)).T
        gauss = np.exp(-dst)*amp + np.random.random_sample(size=(x_range, y_range))*noise_amp

        # repeat this 2D array so it can have (almost) the same size as a real image from the pixelfly camera
        gauss = np.repeat(gauss, round(xmax/x_range), axis=0)
        gauss = np.repeat(gauss, round(ymax/y_range), axis=1)

        return gauss