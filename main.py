import re
import sys
import h5py
import time
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import configparser
import PyQt5
import pyqtgraph as pg
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
from collections import deque
import pco
import qdarkstyle # see https://github.com/ColinDuquesnoy/QDarkStyleSheet

# use "cam.sdk.get_camera_description()" to check camera specs:
# sensor type: 16
# sensor subtype: 0
# max. horizontal resolution standard: 1392
# max. vertical resolution standard: 1040
# max. horizontal resolution extended: 800
# max. vertical resolution extended: 600
# dynamic: 14
# max. binning horizontal: 4
# binning horizontal stepping: 0
# max. binning vert: 4
# binning vert stepping: 0
# roi hor steps: 0
# roi vert steps: 0
# number adcs: 1
# min size horz: 64
# pixel rate: [12000000, 24000000, 0, 0]
# conversion factor: [100, 150, 0, 0]
# cooling setpoints: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# soft roi hor steps: 0
# soft roi vert steps: 0
# ir: 1
# min size vert: 16
# Min Delay DESC: 0
# Max Delay DESC: 0
# Min Delay StepDESC: 100
# Min Expos DESC: 1000
# Max Expos DESC: 60000
# Min Expos Step DESC: 1000
# Min Delay IR DESC: 0
# Max Delay IR DESC: 0
# Min Expos IR DESC: 5000000
# Max ExposIR DESC: 60000
# Time Table DESC: 0
# wDoubleImageDESC: 1
# Min Cool Set DESC: 0
# Max Cool Set DESC: 0
# Default Cool Set DESC: 0
# Power Down Mode DESC: 0
# Offset Regulation DESC: 0
# Color Pattern DESC: 0
# Pattern Type DESC: 0
# Num Cooling Setpoints: 0
# dwGeneralCapsDESC1: 4800
# dwGeneralCapsDESC2: 0
# ext sync frequency: [0, 0, 0, 0]
# dwGeneralCapsDESC3: 0
# dwGeneralCapsDESC4: 0

sensor_format_options = {"standard (1392*1040)": ("standard", (0, 1391), (0, 1039)),
                        "extended (800*600)": ("extended", (0, 799), (0, 599))}
sensor_format_default = list(sensor_format_options.keys())[0]
clock_rate_options = {"12 MHz": 12000000, "24 MHz": 24000000}
clock_rate_default = list(clock_rate_options.keys())[0]
conv_factor_options = {"1": 100, "1.5": 150} # 100/gain, or electrons/count*100
conv_factor_default = list(conv_factor_options.keys())[0]
trigger_mode_options = {"auto sequence": "auto sequence", "software trigger": "software trigger",
                        "external/software trigger": "external exposure start & software trigger"}
trigger_mode_default = list(trigger_mode_options.keys())[2]
trigger_source_options = ["software", "external TTL"]
trigger_source_default = trigger_source_options[1]
expo_unit_options = {"ms": 1e-3, "us": 1e-6} # unit "ns" is not implemented bacause the min expo time is 1000 ns = 1 us
expo_unit_default = list(expo_unit_options.keys())[0]
expo_time_default = "10" # in the unit of expo_unit_default
expo_min = 1000e-9 # Min Expos DESC in ns
expo_max = 60000e-3 # Max Expos DESC in ms
expo_decimals = 6 # converted from Min Expos Step DESC in ns
binning_max = 4 # same for both horizontal and vertical
binning_default = (1, 1) # (horizontal, vertical)
gaussian_fit_default = False
img_auto_save_dafault = True

# steal colormap datat from matplotlib
def steal_colormap(colorname="viridis", lut=12):
    color = cm.get_cmap(colorname, lut)
    colordata = color(range(lut)) # (r, g, b, a=opacity)
    colordata_reform = []
    for i in range(lut):
        l = [i/lut, tuple(colordata[i]*255)]
        colordata_reform.append(tuple(l))

    return colordata_reform

def fake_data(x_range=1392, y_range=1040, x_center=700, y_center=500, x_err=100, y_err=50, amp=20):
    x, y = np.meshgrid(np.arange(x_range), np.arange(y_range))
    dst = np.sqrt((x-x_center)**2/(2*x_err**2)+(y-y_center)**2/2/(2*y_err**2)).T
    gauss = np.exp(-dst)*amp + np.random.normal(size=(x_range, y_range))

    return gauss

class Scrollarea(qt.QGroupBox):
    def __init__(self, parent, label="", type="grid"):
        super().__init__()
        self.parent = parent
        self.setTitle(label)
        outer_layout = qt.QGridLayout()
        self.setLayout(outer_layout)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(0x10)
        outer_layout.addWidget(scroll)

        box = qt.QWidget()
        scroll.setWidget(box)
        if type == "form":
            self.frame = qt.QFormLayout()
        elif type == "grid":
            self.frame = qt.QGridLayout()
        else:
            self.frame = qt.QGridLayout()
            print("Frame type not supported!")

        box.setLayout(self.frame)


class pixelfly:
    def __init__(self, parent):
        self.parent = parent

        # try:
        #     self.cam = pco.Camera(interface='USB 2.0')
        # except Exception as err:
        #     logging.error(traceback.format_exc())
        #     logging.error("Can't open camera")
        #     return
        #
        # self.set_sensor_format(sensor_format_default)
        # self.set_clock_rate(clock_rate_default)
        # self.set_conv_factor(conv_factor_default)
        # self.set_trigger_mode(trigger_mode_default)
        # self.trigger_source = trigger_source_default
        # self.set_expo_time(expo_time_default, expo_unit_default)
        # self.set_binning(binning_default[0], binning_default[1])
        # self.gaussian_fit = gaussian_fit_default
        # self.img_save = img_auto_save_default

    def set_sensor_format(self, arg):
        self.cam.sdk.set_sensor_format(sensor_format_options[arg][0])
        self.cam.sdk.arm_camera()

    def set_clock_rate(self, arg):
        self.cam.configuration = {"pixel rate": clock_rate_options[arg]}

    def set_conv_factor(self, arg):
        self.cam.sdk.set_conversion_factor(conv_factor_options[arg])
        self.cam.sdk.arm_camera()

    def set_trigger_mode(self, arg):
        self.cam.configuration = {"trigger": trigger_mode_options[arg]}

    def set_trigger_source(self, text, checked):
        if checked:
            self.trigger_source = text

    def set_expo_time(self, time, unit):
        # exposure time sanity check
        try:
            expo_time = float(time)*expo_unit_options[unit]
        except ValueError as err:
            logging.warning(traceback.format_exc())
            logging.warning("Exposure time invalid!")
            return

        expo_time_round = round(expo_time, expo_decimals)
        if expo_time_round < expo_min:
            expo_time_round = expo_min
        elif expo_time_round > expo_max:
            expo_time_round = expo_max
        self.cam.configuration = {'exposure time': expo_time_round}

        d = int(expo_decimals+np.log10(expo_unit_options[unit]))
        if d:
            t = round(expo_time_round/expo_unit_options[unit], d)
            t = f"{t}"
        else:
            t = "{:d}".format(round(expo_time_round/expo_unit_options[unit]))
        return t

    def set_binning(self, bin_h, bin_v):
        self.cam.configuration = {'binning': (int(bin_h), int(bin_v))}

    def set_gauss_fit(self, state):
        self.gaussian_fit = state

    def set_img_save(self, state):
        self.img_save = state

    def load_configs(self):
        pass


class Cam_Control(qt.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMaximumWidth(400)
        self.frame = qt.QVBoxLayout()
        self.setLayout(self.frame)
        self.frame.setSpacing(0)

        self.place_record_elements()
        self.place_control_elements()
        self.place_indicator_elements()

    def place_record_elements(self):
        record_box = Scrollarea(self, label="Recording", type="grid")
        record_box.setMaximumHeight(100)
        self.frame.addWidget(record_box)

        self.record_bt = qt.QPushButton("Record")
        self.record_bt.clicked[bool].connect(lambda val: self.start())
        record_box.frame.addWidget(self.record_bt, 0, 0)

        self.scan_bt = qt.QPushButton("Scan")
        self.scan_bt.clicked[bool].connect(lambda val: self.scan())
        record_box.frame.addWidget(self.scan_bt, 0, 1)

        self.stop_bt = qt.QPushButton("Stop")
        self.stop_bt.clicked[bool].connect(lambda val: self.stop())
        record_box.frame.addWidget(self.stop_bt, 0, 2)
        self.stop_bt.setEnabled(False)

    def place_control_elements(self):
        self.control_elem_box = Scrollarea(self, label="Camera Control", type="form")
        self.frame.addWidget(self.control_elem_box)

        self.sensor_format_cb = qt.QComboBox()
        self.sensor_format_cb.setMaximumWidth(200)
        self.sensor_format_cb.setMaximumHeight(20)
        for i in range(len(sensor_format_options)):
            self.sensor_format_cb.addItem(list(sensor_format_options.keys())[i])
        self.sensor_format_cb.setCurrentText(sensor_format_default)
        self.sensor_format_cb.activated[str].connect(lambda val: self.set_sensor_format(val))
        self.control_elem_box.frame.addRow("Sensor size:", self.sensor_format_cb)

        self.clock_rate_cb = qt.QComboBox()
        self.clock_rate_cb.setMaximumWidth(200)
        self.clock_rate_cb.setMaximumHeight(20)
        for i in range(len(clock_rate_options)):
            self.clock_rate_cb.addItem(list(clock_rate_options.keys())[i])
        self.clock_rate_cb.setCurrentText(clock_rate_default)
        self.clock_rate_cb.activated[str].connect(lambda val: self.parent.device.set_clock_rate(val))
        self.control_elem_box.frame.addRow("Clock rate:", self.clock_rate_cb)

        self.conv_factor_cb = qt.QComboBox()
        self.conv_factor_cb.setMaximumWidth(200)
        self.conv_factor_cb.setMaximumHeight(20)
        for i in range(len(conv_factor_options)):
            self.conv_factor_cb.addItem(list(conv_factor_options.keys())[i])
        self.conv_factor_cb.setCurrentText(conv_factor_default)
        self.conv_factor_cb.activated[str].connect(lambda val: self.parent.device.set_conv_factor(val))
        self.control_elem_box.frame.addRow("Conversion factor:", self.conv_factor_cb)

        self.trigger_mode_cb = qt.QComboBox()
        self.trigger_mode_cb.setMaximumWidth(200)
        self.trigger_mode_cb.setMaximumHeight(20)
        for i in range(len(trigger_mode_options)):
            self.trigger_mode_cb.addItem(list(trigger_mode_options.keys())[i])
        self.trigger_mode_cb.setCurrentText(trigger_mode_default)
        self.trigger_mode_cb.activated[str].connect(lambda val: self.parent.device.set_trigger_mode(val))
        self.control_elem_box.frame.addRow("Trigger mode:", self.trigger_mode_cb)

        self.trig_source_rblist = []
        trig_bg = qt.QButtonGroup(self.parent)
        trig_box = qt.QWidget()
        trig_box.setMaximumWidth(200)
        trig_layout = qt.QHBoxLayout()
        trig_layout.setContentsMargins(0,0,0,0)
        trig_box.setLayout(trig_layout)
        for i in range(len(trigger_source_options)):
            trig_source_rb = qt.QRadioButton(trigger_source_options[i])
            trig_source_rb.setChecked(True if trigger_source_default == trigger_source_options[i] else False)
            trig_source_rb.toggled[bool].connect(lambda val, rb=trig_source_rb: self.parent.device.set_trigger_source(rb.text(), val))
            self.trig_source_rblist.append(trig_source_rb)
            trig_bg.addButton(trig_source_rb)
            trig_layout.addWidget(trig_source_rb)
        self.control_elem_box.frame.addRow("Trigger source:", trig_box)

        self.expo_le = qt.QLineEdit() # try qt.QDoubleSpinBox() ?
        self.expo_le.setText(expo_time_default)
        self.expo_unit_cb = qt.QComboBox()
        self.expo_unit_cb.setMaximumHeight(30)
        for i in range(len(expo_unit_options)):
            self.expo_unit_cb.addItem(list(expo_unit_options.keys())[i])
        self.expo_unit_cb.setCurrentText(expo_unit_default)
        self.expo_le.editingFinished.connect(lambda le=self.expo_le, cb=self.expo_unit_cb:
                                            self.set_expo_time(le.text(), cb.currentText()))
        self.expo_unit_cb.activated[str].connect(lambda val, le=self.expo_le: self.set_expo_time(le.text(), val))
        expo_box = qt.QWidget()
        expo_box.setMaximumWidth(200)
        expo_layout = qt.QHBoxLayout()
        expo_layout.setContentsMargins(0,0,0,0)
        expo_box.setLayout(expo_layout)
        expo_layout.addWidget(self.expo_le)
        expo_layout.addWidget(self.expo_unit_cb)
        self.control_elem_box.frame.addRow("Exposure time:", expo_box)

        self.bin_hori_sb = qt.QSpinBox()
        self.bin_hori_sb.setRange(1, binning_max)
        self.bin_hori_sb.setValue(binning_default[0])
        self.bin_vert_sb = qt.QSpinBox()
        self.bin_vert_sb.setRange(1, binning_max)
        self.bin_vert_sb.setValue(binning_default[1])
        self.bin_hori_sb.valueChanged[int].connect(lambda val, sb=self.bin_vert_sb: self.parent.device.set_binning(val, sb.value()))
        self.bin_vert_sb.valueChanged[int].connect(lambda val, sb=self.bin_hori_sb: self.parent.device.set_binning(sb.value(), val))
        bin_box = qt.QWidget()
        bin_box.setMaximumWidth(200)
        bin_layout = qt.QHBoxLayout()
        bin_layout.setContentsMargins(0,0,0,0)
        bin_box.setLayout(bin_layout)
        bin_layout.addWidget(self.bin_hori_sb)
        bin_layout.addWidget(self.bin_vert_sb)
        self.control_elem_box.frame.addRow("Binning H x V:", bin_box)

        self.load_settings_bt = qt.QPushButton("load settings")
        self.load_settings_bt.clicked[bool].connect(lambda val: self.load_settings())
        self.control_elem_box.frame.addRow("Load settings:", self.load_settings_bt)

        self.save_settings_bt = qt.QPushButton("save settings")
        self.save_settings_bt.clicked[bool].connect(lambda val: self.save_settings())
        self.control_elem_box.frame.addRow("Save settings:", self.save_settings_bt)

    def place_indicator_elements(self):
        indicator_box = Scrollarea(self, label="Indicators", type="form")
        self.frame.addWidget(indicator_box)



        # self.cc_err_mean = qt.QLabel()
        # self.cc_err_mean.setText("0")
        # self.cc_err_mean.setStyleSheet("background-color: gray; font: 20pt")
        # indicator_box.frame.addRow("Camera count error of mean:", self.cc_err_mean)

        indicator_box.frame.addRow("------------------", qt.QWidget())

        self.x_mean = qt.QLabel()
        self.x_mean.setText("0")
        self.x_mean.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (x mean):", self.x_mean)

        self.x_stand_dev = qt.QLabel()
        self.x_stand_dev.setText("0")
        self.x_stand_dev.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (x stan. dev.):", self.x_stand_dev)

        self.y_mean = qt.QLabel()
        self.y_mean.setText("0")
        self.y_mean.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (y mean):", self.y_mean)

        self.y_stand_dev = qt.QLabel()
        self.y_stand_dev.setText("0")
        self.y_stand_dev.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (y stan. dev.):", self.y_stand_dev)

        self.amp = qt.QLabel()
        self.amp.setText("0")
        self.amp.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (amp.):", self.amp)

        self.offset = qt.QLabel()
        self.offset.setText("0")
        self.offset.setStyleSheet("background-color: gray;")
        indicator_box.frame.addRow("2D gaussian fit (offset):", self.offset)

    def start(self):
        self.record_bt.setEnabled(False)
        self.scan_bt.setEnabled(False)
        self.stop_bt.setEnabled(True)
        self.control_elem_box.setEnabled(False)

    def scan(self):
        self.scan_bt.setEnabled(False)
        self.record_bt.setEnabled(False)
        self.stop_bt.setEnabled(True)
        self.control_elem_box.setEnabled(False)

    def stop(self):
        self.stop_bt.setEnabled(False)
        self.record_bt.setEnabled(True)
        self.scan_bt.setEnabled(True)
        self.control_elem_box.setEnabled(True)

    def set_sensor_format(self, val):
        self.parent.device.set_sensor_format(val)
        # to-do: modify plots range spinboxes

    def set_expo_time(self, time, unit):
        t = self.parent.device.set_expo_time(time, unit)
        self.expo_le.setText(t)

    def load_settings(self):
        pass

    def save_settings(self):
        pass


class Img_Control(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Image Control", type="grid")

        self.setMaximumHeight(200)
        self.place_range_control()
        self.place_num_image()
        self.place_gauss_fit()
        self.place_auto_save()
        self.place_camera_count()

    def place_range_control(self):
        self.frame.addWidget(qt.QLabel("X range:"), 0, 0)
        x_min = sensor_format_options[sensor_format_default][1][0]
        x_max = sensor_format_options[sensor_format_default][1][1]
        self.x_min_sb = qt.QSpinBox()
        self.x_min_sb.setRange(x_min, x_max)
        self.x_min_sb.setValue(x_min)
        self.x_max_sb = qt.QSpinBox()
        self.x_max_sb.setRange(x_min, x_max)
        self.x_max_sb.setValue(x_max)
        self.x_min_sb.valueChanged[int].connect(lambda val, text='xmin', sb=self.x_max_sb:
                                                self.range_change(text, val, sb))
        self.x_max_sb.valueChanged[int].connect(lambda val, text='xmax', sb=self.x_min_sb:
                                                self.range_change(text, val, sb))

        x_range_box = qt.QWidget()
        self.frame.addWidget(x_range_box, 0, 1)
        x_range_layout = qt.QHBoxLayout()
        x_range_layout.setContentsMargins(0,0,0,0)
        x_range_box.setLayout(x_range_layout)
        x_range_layout.addWidget(self.x_min_sb)
        x_range_layout.addWidget(self.x_max_sb)

        self.frame.addWidget(qt.QLabel("Y range:"), 1, 0)
        y_min = sensor_format_options[sensor_format_default][2][0]
        y_max = sensor_format_options[sensor_format_default][2][1]
        self.y_min_sb = qt.QSpinBox()
        self.y_min_sb.setRange(y_min, y_max)
        self.y_min_sb.setValue(y_min)
        self.y_max_sb = qt.QSpinBox()
        self.y_max_sb.setRange(y_min, y_max)
        self.y_max_sb.setValue(y_max)
        self.y_min_sb.valueChanged[int].connect(lambda val, text='ymin', sb=self.y_max_sb:
                                                self.range_change(text, val, sb))
        self.y_max_sb.valueChanged[int].connect(lambda val, text='ymax', sb=self.y_min_sb:
                                                self.range_change(text, val, sb))

        y_range_box = qt.QWidget()
        self.frame.addWidget(y_range_box, 1, 1)
        y_range_layout = qt.QHBoxLayout()
        y_range_layout.setContentsMargins(0,0,0,0)
        y_range_box.setLayout(y_range_layout)
        y_range_layout.addWidget(self.y_min_sb)
        y_range_layout.addWidget(self.y_max_sb)

    def place_num_image(self):
        self.frame.addWidget(qt.QLabel("Num of recorded images: "), 0, 2)
        self.num_image = qt.QLabel()
        self.num_image.setText("0")
        self.num_image.setStyleSheet("background-color: gray;")
        self.frame.addWidget(self.num_image, 0, 3)

        self.frame.addWidget(qt.QLabel("Image width x height:"), 1, 2)
        self.image_width = qt.QLabel()
        self.image_width.setText("0")
        self.image_width.setStyleSheet("background-color: gray;")
        self.image_height = qt.QLabel()
        self.image_height.setText("0")
        self.image_height.setStyleSheet("background-color: gray;")
        image_size_box = qt.QWidget()
        image_size_layout = qt.QHBoxLayout()
        image_size_layout.setContentsMargins(0,0,0,0)
        image_size_box.setLayout(image_size_layout)
        image_size_layout.addWidget(self.image_width)
        image_size_layout.addWidget(self.image_height)
        self.frame.addWidget(image_size_box, 1, 3)

    def place_gauss_fit(self):
        self.frame.addWidget(qt.QLabel("2D gaussian fit:"), 0, 4)
        self.gauss_fit_chb = qt.QCheckBox()
        self.gauss_fit_chb.setTristate(False)
        self.gauss_fit_chb.setCheckState(0 if gaussian_fit_default in [False, 0, "False", "false"] else 2)
        self.gauss_fit_chb.setStyleSheet("QCheckBox::indicator {width: 18px; height: 18px;}")
        self.gauss_fit_chb.stateChanged[int].connect(lambda state: self.parent.device.set_gauss_fit(state))
        self.frame.addWidget(self.gauss_fit_chb, 0, 5)

    def place_auto_save(self):
        self.frame.addWidget(qt.QLabel("Image auto save:"), 1, 4)
        self.img_save_chb = qt.QCheckBox()
        self.img_save_chb.setTristate(False)
        self.img_save_chb.setCheckState(0 if img_auto_save_dafault in [False, 0, "False", "false"] else 2)
        self.img_save_chb.setStyleSheet("QCheckBox::indicator {width: 18px; height: 18px;}")
        self.img_save_chb.stateChanged[int].connect(lambda state: self.parent.device.set_img_save(state))
        self.frame.addWidget(self.img_save_chb, 1, 5)

    def place_camera_count(self):
        self.frame.addWidget(qt.QLabel("Camera count:"), 2, 0, 1, 2)
        self.camera_count = qt.QLabel()
        self.camera_count.setMaximumWidth(200)
        self.camera_count.setText("0")
        self.camera_count.setStyleSheet("background-color: gray; font: 20pt")
        self.frame.addWidget(self.camera_count, 3, 0, 1, 2)

        self.frame.addWidget(qt.QLabel("Camera count mean:"), 2, 2, 1, 2)
        self.camera_count_mean = qt.QLabel()
        self.camera_count_mean.setText("0")
        self.camera_count_mean.setStyleSheet("background-color: gray; font: 20pt")
        self.frame.addWidget(self.camera_count_mean, 3, 2, 1, 2)

        self.frame.addWidget(qt.QLabel("Camera count error of mean:"), 2, 4, 1, 2)
        self.camera_count_err_mean = qt.QLabel()
        self.camera_count_err_mean.setText("0")
        self.camera_count_err_mean.setStyleSheet("background-color: gray; font: 20pt")
        self.frame.addWidget(self.camera_count_err_mean, 3, 4, 1, 2)

    def range_change(self):
        pass


class Image(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Images", type="grid")

        self.place_plots()

    def place_plots(self):
        self.colormap = steal_colormap()

        self.bg_graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.frame.addWidget(self.bg_graphlayout)

        self.bg_plot = self.bg_graphlayout.addPlot(title="Background")
        self.bg_img = pg.ImageItem()
        self.bg_plot.addItem(self.bg_img)

        self.bg_hist = pg.HistogramLUTItem()
        self.bg_hist.setImageItem(self.bg_img)
        self.bg_graphlayout.addItem(self.bg_hist)
        self.bg_hist.gradient.restoreState({'mode': 'rgb', 'ticks': self.colormap})

        data = fake_data(x_range=20, y_range=20, x_center=10, y_center=10, x_err=5, y_err=3)
        self.bg_img.setImage(data)

class CameraGUI(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('pco.pixelfly usb')
        self.setStyleSheet("QWidget{font: 10pt;}")

        self.device = pixelfly(self)
        self.cam_control = Cam_Control(self)
        self.img_control = Img_Control(self)
        self.image = Image(self)

        self.splitter1 = qt.QSplitter()
        self.splitter2 = qt.QSplitter()
        self.splitter1.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.splitter2.setOrientation(PyQt5.QtCore.Qt.Vertical)
        self.setCentralWidget(self.splitter1)
        self.splitter1.addWidget(self.splitter2)
        self.splitter1.addWidget(self.cam_control)
        self.splitter2.addWidget(self.img_control)
        self.splitter2.addWidget(self.image)

        self.resize(1600, 900)
        self.show()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = CameraGUI()
    app.exec_()
    main_window.device.cam.close()
    sys.exit(0)
