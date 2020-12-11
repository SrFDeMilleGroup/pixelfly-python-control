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
img_auto_save_default = True
num_img_to_take_default = 10

# steal colormap datat from matplotlib
def steal_colormap(colorname="viridis", lut=12):
    color = cm.get_cmap(colorname, lut)
    colordata = color(range(lut)) # (r, g, b, a=opacity)
    colordata_reform = []
    for i in range(lut):
        l = [i/lut, tuple(colordata[i]*255)]
        colordata_reform.append(tuple(l))

    return colordata_reform

def fake_data(x_range=30, y_range=20, x_center=20, y_center=12, x_err=5, y_err=5, amp=10000):
    x, y = np.meshgrid(np.arange(x_range), np.arange(y_range))
    dst = np.sqrt((x-x_center)**2/(2*x_err**2)+(y-y_center)**2/2/(2*y_err**2)).T
    gauss = np.exp(-dst)*amp + np.random.random_sample(size=(x_range, y_range))*amp/20
    gauss = gauss.astype("uint16")

    return gauss

class Scrollarea(qt.QGroupBox):
    def __init__(self, parent, label="", type="grid"):
        super().__init__()
        self.parent = parent
        self.setTitle(label)
        outer_layout = qt.QGridLayout()
        outer_layout.setContentsMargins(0,0,0,0)
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
        elif type == "vbox":
            self.frame = qt.QVBoxLayout()
        elif type == "hbox":
            self.frame = qt.QHBoxLayout()
        else:
            self.frame = qt.QGridLayout()
            print("Frame type not supported!")

        box.setLayout(self.frame)


class CamThread(PyQt5.QtCore.QThread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        self.counter = 0
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(700)

    def update(self):
        data = np.random.randint(low=0, high=1000, size=(200, 200), dtype=np.uint16)
        self.parent.image_win.imgs_dict["Background"].setImage(data)
        self.parent.image_win.x_plot.setData(np.sum(data, axis=1))
        self.parent.image_win.y_plot.setData(np.sum(data, axis=0))

        self.counter += 1
        if self.counter == 10:
            self.timer.stop()


class pixelfly:
    def __init__(self, parent):
        self.parent = parent
        self.counter = 0

        try:
            self.cam = pco.Camera(interface='USB 2.0')
        except Exception as err:
            logging.error(traceback.format_exc())
            logging.error("Can't open camera")
            return

        self.set_sensor_format(sensor_format_default)
        self.set_clock_rate(clock_rate_default)
        self.set_conv_factor(conv_factor_default)
        self.set_trigger_mode(trigger_mode_default)
        self.trigger_source = trigger_source_default
        self.set_expo_time(expo_time_default, expo_unit_default)
        self.set_binning(binning_default[0], binning_default[1])
        self.gaussian_fit = gaussian_fit_default
        self.img_save = img_auto_save_default
        self.num_img_to_take = num_img_to_take_default

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

    def set_num_img(self, val):
        self.num_img_to_take = val

    def recording(self):
        # data = fake_data(x_range=2000, y_range=3000, x_center=1400, y_center=1500, x_err=500, y_err=500)
        self.counter += 1
        while True:
            if self.cam.rec.get_status()['dwProcImgCount'] >=self.counter:
                # print(self.cam.rec.get_status()['dwProcImgCount'])
                break
            time.sleep(0.001)
        image, meta = self.cam.image(image_number=0xFFFFFFFF) # readout the last image
        data = np.flip(image.T, 1)
        self.parent.image_win.imgs_dict["Background"].setImage(data)
        self.parent.image_win.x_plot.setData(np.sum(data, axis=1))
        self.parent.image_win.y_plot.setData(np.sum(data, axis=0))


        # print(f"{self.counter}: "+"{:.5f}".format(time.time()-self.parent.control.last_time))
        if self.counter == self.num_img_to_take:
            self.parent.app.processEvents()
            self.parent.control.timer.stop()
            self.cam.stop()
            self.counter = 0

    def load_configs(self):
        pass


class Control(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="", type="vbox")
        self.setMaximumWidth(400)
        self.frame.setContentsMargins(0,0,0,0)

        self.place_recording()
        self.place_image_control()
        self.place_cam_control()

    def place_recording(self):
        record_box = qt.QGroupBox("Recording")
        record_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        record_box.setMaximumHeight(180)
        record_frame = qt.QGridLayout()
        record_box.setLayout(record_frame)
        self.frame.addWidget(record_box)

        self.record_bt = qt.QPushButton("Record")
        self.record_bt.clicked[bool].connect(lambda val: self.record())
        record_frame.addWidget(self.record_bt, 0, 0)

        self.scan_bt = qt.QPushButton("Scan")
        self.scan_bt.clicked[bool].connect(lambda val: self.scan())
        record_frame.addWidget(self.scan_bt, 0, 1)

        self.stop_bt = qt.QPushButton("Stop")
        self.stop_bt.clicked[bool].connect(lambda val: self.stop())
        record_frame.addWidget(self.stop_bt, 0, 2)
        self.stop_bt.setEnabled(False)

        record_frame.addWidget(qt.QLabel("Camera count:"), 1, 0, 1, 1)
        self.camera_count = qt.QLabel()
        self.camera_count.setText("0")
        self.camera_count.setStyleSheet("background-color: gray; font: 20pt")
        record_frame.addWidget(self.camera_count, 1, 1, 1, 2)

        record_frame.addWidget(qt.QLabel("Cam. mean:"), 2, 0, 1, 1)
        self.camera_count_mean = qt.QLabel()
        self.camera_count_mean.setText("0")
        self.camera_count_mean.setStyleSheet("background-color: gray; font: 20pt")
        record_frame.addWidget(self.camera_count_mean, 2, 1, 1, 2)

        record_frame.addWidget(qt.QLabel("Cam. error.:"), 3, 0, 1, 1)
        self.camera_count_err_mean = qt.QLabel()
        self.camera_count_err_mean.setText("0")
        self.camera_count_err_mean.setStyleSheet("background-color: gray; font: 20pt")
        record_frame.addWidget(self.camera_count_err_mean, 3, 1, 1, 2)

    def place_image_control(self):
        img_ctrl_box = qt.QGroupBox("Image Control")
        img_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        img_ctrl_frame = qt.QFormLayout()
        img_ctrl_box.setLayout(img_ctrl_frame)
        self.frame.addWidget(img_ctrl_box)

        num_img_to_take = qt.QSpinBox()
        num_img_to_take.setRange(1, 1000000)
        num_img_to_take.setValue(num_img_to_take_default)
        num_img_to_take.valueChanged[int].connect(lambda val: self.parent.device.set_num_img(val))
        img_ctrl_frame.addRow("Num of image to take:", num_img_to_take)

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
        x_range_layout = qt.QHBoxLayout()
        x_range_layout.setContentsMargins(0,0,0,0)
        x_range_box.setLayout(x_range_layout)
        x_range_layout.addWidget(self.x_min_sb)
        x_range_layout.addWidget(self.x_max_sb)
        img_ctrl_frame.addRow("X range:", x_range_box)

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
        y_range_layout = qt.QHBoxLayout()
        y_range_layout.setContentsMargins(0,0,0,0)
        y_range_box.setLayout(y_range_layout)
        y_range_layout.addWidget(self.y_min_sb)
        y_range_layout.addWidget(self.y_max_sb)
        img_ctrl_frame.addRow("Y range:", y_range_box)

        self.num_image = qt.QLabel()
        self.num_image.setText("0")
        self.num_image.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("Num of recorded images:", self.num_image)

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
        img_ctrl_frame.addRow("Image width x height:", image_size_box)

        self.gauss_fit_chb = qt.QCheckBox()
        self.gauss_fit_chb.setTristate(False)
        self.gauss_fit_chb.setCheckState(0 if gaussian_fit_default in [False, 0, "False", "false"] else 2)
        self.gauss_fit_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gauss_fit_chb.stateChanged[int].connect(lambda state: self.parent.device.set_gauss_fit(state))
        img_ctrl_frame.addRow("2D gaussian fit:", self.gauss_fit_chb)

        self.img_save_chb = qt.QCheckBox()
        self.img_save_chb.setTristate(False)
        self.img_save_chb.setCheckState(0 if img_auto_save_default in [False, 0, "False", "false"] else 2)
        self.img_save_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.img_save_chb.stateChanged[int].connect(lambda state: self.parent.device.set_img_save(state))
        img_ctrl_frame.addRow("Image auto save:", self.img_save_chb)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        self.x_mean = qt.QLabel()
        self.x_mean.setText("0")
        self.x_mean.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (x mean):", self.x_mean)

        self.x_stand_dev = qt.QLabel()
        self.x_stand_dev.setText("0")
        self.x_stand_dev.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (x stan. dev.):", self.x_stand_dev)

        self.y_mean = qt.QLabel()
        self.y_mean.setText("0")
        self.y_mean.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (y mean):", self.y_mean)

        self.y_stand_dev = qt.QLabel()
        self.y_stand_dev.setText("0")
        self.y_stand_dev.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (y stan. dev.):", self.y_stand_dev)

        self.amp = qt.QLabel()
        self.amp.setText("0")
        self.amp.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (amp.):", self.amp)

        self.offset = qt.QLabel()
        self.offset.setText("0")
        self.offset.setStyleSheet("background-color: gray;")
        img_ctrl_frame.addRow("2D gaussian fit (offset):", self.offset)

    def place_cam_control(self):
        self.cam_ctrl_box = qt.QGroupBox("Camera Control")
        self.cam_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        cam_ctrl_frame = qt.QFormLayout()
        self.cam_ctrl_box.setLayout(cam_ctrl_frame)
        self.frame.addWidget(self.cam_ctrl_box)

        self.sensor_format_cb = qt.QComboBox()
        self.sensor_format_cb.setMaximumWidth(200)
        self.sensor_format_cb.setMaximumHeight(20)
        for i in range(len(sensor_format_options)):
            self.sensor_format_cb.addItem(list(sensor_format_options.keys())[i])
        self.sensor_format_cb.setCurrentText(sensor_format_default)
        self.sensor_format_cb.activated[str].connect(lambda val: self.set_sensor_format(val))
        cam_ctrl_frame.addRow("Sensor size:", self.sensor_format_cb)

        self.clock_rate_cb = qt.QComboBox()
        self.clock_rate_cb.setMaximumWidth(200)
        self.clock_rate_cb.setMaximumHeight(20)
        for i in range(len(clock_rate_options)):
            self.clock_rate_cb.addItem(list(clock_rate_options.keys())[i])
        self.clock_rate_cb.setCurrentText(clock_rate_default)
        self.clock_rate_cb.activated[str].connect(lambda val: self.parent.device.set_clock_rate(val))
        cam_ctrl_frame.addRow("Clock rate:", self.clock_rate_cb)

        self.conv_factor_cb = qt.QComboBox()
        self.conv_factor_cb.setMaximumWidth(200)
        self.conv_factor_cb.setMaximumHeight(20)
        for i in range(len(conv_factor_options)):
            self.conv_factor_cb.addItem(list(conv_factor_options.keys())[i])
        self.conv_factor_cb.setCurrentText(conv_factor_default)
        self.conv_factor_cb.activated[str].connect(lambda val: self.parent.device.set_conv_factor(val))
        cam_ctrl_frame.addRow("Conversion factor:", self.conv_factor_cb)

        self.trigger_mode_cb = qt.QComboBox()
        self.trigger_mode_cb.setMaximumWidth(200)
        self.trigger_mode_cb.setMaximumHeight(20)
        for i in range(len(trigger_mode_options)):
            self.trigger_mode_cb.addItem(list(trigger_mode_options.keys())[i])
        self.trigger_mode_cb.setCurrentText(trigger_mode_default)
        self.trigger_mode_cb.activated[str].connect(lambda val: self.parent.device.set_trigger_mode(val))
        cam_ctrl_frame.addRow("Trigger mode:", self.trigger_mode_cb)

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
        cam_ctrl_frame.addRow("Trigger source:", trig_box)

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
        cam_ctrl_frame.addRow("Exposure time:", expo_box)

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
        cam_ctrl_frame.addRow("Binning H x V:", bin_box)

        self.load_settings_bt = qt.QPushButton("load settings")
        self.load_settings_bt.setMaximumWidth(200)
        self.load_settings_bt.clicked[bool].connect(lambda val: self.load_settings())
        cam_ctrl_frame.addRow("Load settings:", self.load_settings_bt)

        self.save_settings_bt = qt.QPushButton("save settings")
        self.save_settings_bt.setMaximumWidth(200)
        self.save_settings_bt.clicked[bool].connect(lambda val: self.save_settings())
        cam_ctrl_frame.addRow("Save settings:", self.save_settings_bt)

    def record(self):
        self.record_bt.setEnabled(False)
        self.scan_bt.setEnabled(False)
        self.stop_bt.setEnabled(True)
        self.cam_ctrl_box.setEnabled(False)

        # self.t = CamThread(self.parent)
        # self.t.start()
        self.parent.device.cam.record(number_of_images=5, mode='ring buffer')
        # number_of_images is buffer size in ring buffer mode

        self.timer = PyQt5.QtCore.QTimer()
        self.timer.timeout.connect(self.parent.device.recording)
        self.timer.start(0)
        self.last_time = time.time()

    def scan(self):
        self.scan_bt.setEnabled(False)
        self.record_bt.setEnabled(False)
        self.stop_bt.setEnabled(True)
        self.cam_ctrl_box.setEnabled(False)

    def stop(self):
        self.running = False

        self.stop_bt.setEnabled(False)
        self.record_bt.setEnabled(True)
        self.scan_bt.setEnabled(True)
        self.cam_ctrl_box.setEnabled(True)

    def range_change(self):
        pass

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


class ImageWin(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="Images", type="grid")
        self.colormap = steal_colormap()
        self.frame.setColumnStretch(0,2)
        self.frame.setColumnStretch(1,1)
        self.frame.setRowStretch(0,1)
        self.frame.setRowStretch(1,1)
        self.frame.setRowStretch(2,1)
        self.imgs_dict = {}
        self.imgs_name = ["Background", "Signal", "Signal w/ bg subtraction"]

        self.place_sgn_imgs()
        self.place_axis_plots()
        self.place_ave_image()
        self.place_scan_plot()

    def place_sgn_imgs(self):
        self.img_tab = qt.QTabWidget()
        self.frame.addWidget(self.img_tab, 0, 0, 2, 1)
        for i, name in enumerate(self.imgs_name):
            graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
            self.img_tab.addTab(graphlayout, name)
            plot = graphlayout.addPlot(title=name)
            img = pg.ImageItem()
            plot.addItem(img)

            hist = pg.HistogramLUTItem()
            hist.setImageItem(img)
            graphlayout.addItem(hist)
            hist.gradient.restoreState({'mode': 'rgb', 'ticks': self.colormap})

            self.data = fake_data()
            img.setImage(self.data)

            self.imgs_dict[name] = img

    def place_axis_plots(self):
        tickstyle = {"showValues": False}

        x_data = np.sum(self.data, axis=1)
        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.frame.addWidget(graphlayout, 0, 1, 2, 1)
        x_plot_item = graphlayout.addPlot(title="Camera count v.s. X")
        x_plot_item.showGrid(True, True)
        x_plot_item.setLabel("top")
        # x_plot_item.getAxis("top").setTicks([])
        x_plot_item.getAxis("top").setStyle(**tickstyle)
        x_plot_item.setLabel("right")
        # x_plot_item.getAxis("right").setTicks([])
        x_plot_item.getAxis("right").setStyle(**tickstyle)
        self.x_plot = x_plot_item.plot(x_data)

        graphlayout.nextRow()
        y_data = np.sum(self.data, axis=0)
        y_plot_item = graphlayout.addPlot(title="Camera count v.s. Y")
        y_plot_item.showGrid(True, True)
        y_plot_item.setLabel("top")
        y_plot_item.getAxis("top").setStyle(**tickstyle)
        y_plot_item.setLabel("right")
        y_plot_item.getAxis("right").setStyle(**tickstyle)
        self.y_plot = y_plot_item.plot(y_data)

    def place_ave_image(self):
        graphlayout = pg.GraphicsLayoutWidget(parent=self, border=True)
        self.frame.addWidget(graphlayout, 2, 0)
        plot = graphlayout.addPlot(title="Average image")
        self.ave_img = pg.ImageItem()
        plot.addItem(self.ave_img)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.ave_img)
        graphlayout.addItem(hist)
        hist.gradient.restoreState({'mode': 'rgb', 'ticks': self.colormap})

        data = fake_data()/10
        for i in range(9):
            data += fake_data()/10
        self.ave_img.setImage(data)

    def place_scan_plot(self):
        self.scan_plot_widget = pg.PlotWidget(title="Camera count v.s. Scan param.")
        self.scan_plot_widget.showGrid(True, True)
        fontstyle = {"color": "#919191", "font-size": "11pt"}
        self.scan_plot_widget.setLabel("bottom", "Scan parameter", **fontstyle)
        # tickstyle = {"tickTextOffset": 0, "tickTextHeight": -10, "tickTextWidth": 0}
        # self.scan_plot_widget.getAxis("bottom").setStyle(**tickstyle)
        self.scan_plot = self.scan_plot_widget.plot()
        self.frame.addWidget(self.scan_plot_widget, 2, 1)


class CameraGUI(qt.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle('pco.pixelfly usb')
        self.setStyleSheet("QWidget{font: 10pt;}")
        self.app = app

        self.device = pixelfly(self)
        self.control = Control(self)
        self.image_win = ImageWin(self)

        self.splitter = qt.QSplitter()
        self.splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        self.splitter.addWidget(self.image_win)
        self.splitter.addWidget(self.control)

        self.resize(1600, 900)
        self.show()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = CameraGUI(app)
    app.exec_()
    main_window.device.cam.close()
    sys.exit(0)
