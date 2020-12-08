import re
import sys
import h5py
import time
import logging
import traceback
import numpy as np
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

sensor_format_options = {"standard (1392*1040)": "standard", "extended (800*600)": "extended"}
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

def Scrollarea(label="", type="grid"):
    outer_box = qt.QGroupBox(label)
    outer_layout = qt.QGridLayout()
    outer_box.setLayout(outer_layout)

    scroll = qt.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameStyle(0x10)
    outer_layout.addWidget(scroll)

    box = qt.QWidget()
    scroll.setWidget(box)
    if type == "form":
        inner_layout = qt.QFormLayout()
    elif type == "grid":
        inner_layout = qt.QGridLayout()
    else:
        print("Frame type not supported!")
        return
    box.setLayout(inner_layout)

    return outer_box, inner_layout

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

    def set_sensor_format(self, arg):
        self.cam.sdk.set_sensor_format(sensor_format_options[arg])
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

    def load_configs(self):
        pass


class Controls(qt.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame = qt.QVBoxLayout()
        self.setLayout(self.frame)

        self.place_record_elements()
        self.place_control_elements()
        self.place_indicator_elements()

    def place_record_elements(self):
        outer_box, record_frame = Scrollarea(label="Recording", type="grid")
        self.frame.addWidget(outer_box)

        self.record_bt = qt.QPushButton("Record")
        self.record_bt.clicked[bool].connect(lambda val: self.start())
        record_frame.addWidget(self.record_bt, 0, 0)

        self.scan_bt = qt.QPushButton("Scan")
        self.scan_bt.clicked[bool].connect(lambda val: self.scan())
        record_frame.addWidget(self.scan_bt, 0, 1)

        self.stop_bt = qt.QPushButton("Stop")
        self.stop_bt.clicked[bool].connect(lambda val: self.stop())
        record_frame.addWidget(self.stop_bt, 0, 2)
        self.stop_bt.setEnabled(False)

    def place_control_elements(self):
        self.control_elem_box, control_frame = Scrollarea(label="Controls", type="form")
        self.frame.addWidget(self.control_elem_box)

        self.sensor_format_cb = qt.QComboBox()
        for i in range(len(sensor_format_options)):
            self.sensor_format_cb.addItem(list(sensor_format_options.keys())[i])
        self.sensor_format_cb.setCurrentText(sensor_format_default)
        self.sensor_format_cb.activated[str].connect(lambda val: self.parent.device.set_sensor_format(val))
        control_frame.addRow("Sensor size:", self.sensor_format_cb)

        self.clock_rate_cb = qt.QComboBox()
        for i in range(len(clock_rate_options)):
            self.clock_rate_cb.addItem(list(clock_rate_options.keys())[i])
        self.clock_rate_cb.setCurrentText(clock_rate_default)
        self.clock_rate_cb.activated[str].connect(lambda val: self.parent.device.set_clock_rate(val))
        control_frame.addRow("Clock rate:", self.clock_rate_cb)

        self.conv_factor_cb = qt.QComboBox()
        for i in range(len(conv_factor_options)):
            self.conv_factor_cb.addItem(list(conv_factor_options.keys())[i])
        self.conv_factor_cb.setCurrentText(conv_factor_default)
        self.conv_factor_cb.activated[str].connect(lambda val: self.parent.device.set_conv_factor(val))
        control_frame.addRow("Conversion factor:", self.conv_factor_cb)

        self.trigger_mode_cb = qt.QComboBox()
        for i in range(len(trigger_mode_options)):
            self.trigger_mode_cb.addItem(list(trigger_mode_options.keys())[i])
        self.trigger_mode_cb.setCurrentText(trigger_mode_default)
        self.trigger_mode_cb.activated[str].connect(lambda val: self.parent.device.set_trigger_mode(val))
        control_frame.addRow("Trigger mode:", self.trigger_mode_cb)

        self.trig_source_rblist = []
        trig_bg = qt.QButtonGroup(self.parent)
        trig_box = qt.QWidget()
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
        control_frame.addRow("Trigger source:", trig_box)

        self.expo_le = qt.QLineEdit() # try qt.QDoubleSpinBox() ?
        self.expo_le.setText(expo_time_default)
        self.expo_unit_cb = qt.QComboBox()
        for i in range(len(expo_unit_options)):
            self.expo_unit_cb.addItem(list(expo_unit_options.keys())[i])
        self.expo_unit_cb.setCurrentText(expo_unit_default)
        self.expo_le.editingFinished.connect(lambda le=self.expo_le, cb=self.expo_unit_cb:
                                            self.set_expo_time(le.text(), cb.currentText()))
        self.expo_unit_cb.activated[str].connect(lambda val, le=self.expo_le: self.set_expo_time(le.text(), val))
        expo_box = qt.QWidget()
        expo_layout = qt.QHBoxLayout()
        expo_layout.setContentsMargins(0,0,0,0)
        expo_box.setLayout(expo_layout)
        expo_layout.addWidget(self.expo_le)
        expo_layout.addWidget(self.expo_unit_cb)
        control_frame.addRow("Exposure time:", expo_box)

        self.bin_hori_sb = qt.QSpinBox()
        self.bin_hori_sb.setRange(1, binning_max)
        self.bin_hori_sb.setValue(binning_default[0])
        self.bin_vert_sb = qt.QSpinBox()
        self.bin_vert_sb.setRange(1, binning_max)
        self.bin_vert_sb.setValue(binning_default[1])
        self.bin_hori_sb.valueChanged[int].connect(lambda val, sb=self.bin_vert_sb: self.parent.device.set_binning(val, sb.value()))
        self.bin_vert_sb.valueChanged[int].connect(lambda val, sb=self.bin_hori_sb: self.parent.device.set_binning(sb.value(), val))
        bin_box = qt.QWidget()
        bin_layout = qt.QHBoxLayout()
        bin_layout.setContentsMargins(0,0,0,0)
        bin_box.setLayout(bin_layout)
        bin_layout.addWidget(self.bin_hori_sb)
        bin_layout.addWidget(self.bin_vert_sb)
        control_frame.addRow("Binning H x V:", bin_box)

        self.gauss_fit_chb = qt.QCheckBox()
        self.gauss_fit_chb.setTristate(False)
        self.gauss_fit_chb.setCheckState(0 if gaussian_fit_default in [False, 0, "False", "false"] else 2)
        self.gauss_fit_chb.setStyleSheet("QCheckBox::indicator {width: 25px; height: 25px;}")
        self.gauss_fit_chb.stateChanged[int].connect(lambda state: self.parent.device.set_gauss_fit(state))
        control_frame.addRow("2D gaussian fit:", self.gauss_fit_chb)

        self.load_settings_bt = qt.QPushButton("load settings")
        self.load_settings_bt.clicked[bool].connect(lambda val: self.load_settings())
        control_frame.addRow("Load settings:", self.load_settings_bt)

        self.save_settings_bt = qt.QPushButton("save settings")
        self.save_settings_bt.clicked[bool].connect(lambda val: self.save_settings())
        control_frame.addRow("Save settings:", self.save_settings_bt)

    def place_indicator_elements(self):
        outer_box, indicator_frame = Scrollarea(label="Indicators", type="form")
        self.frame.addWidget(outer_box)

        self.num_image = qt.QLabel()
        self.num_image.setText("0")
        self.num_image.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("Num of recorded images:", self.num_image)

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
        indicator_frame.addRow("Image width x height:", image_size_box)

        self.camera_count = qt.QLabel()
        self.camera_count.setText("0")
        self.camera_count.setStyleSheet("background-color: gray; font: 20pt")
        indicator_frame.addRow("Camera count:", self.camera_count)

        # self.cc_err_mean = qt.QLabel()
        # self.cc_err_mean.setText("0")
        # self.cc_err_mean.setStyleSheet("background-color: gray; font: 20pt")
        # indicator_frame.addRow("Camera count error of mean:", self.cc_err_mean)

        indicator_frame.addRow("------------------", qt.QWidget())

        self.x_mean = qt.QLabel()
        self.x_mean.setText("0")
        self.x_mean.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (x mean):", self.x_mean)

        self.x_stand_dev = qt.QLabel()
        self.x_stand_dev.setText("0")
        self.x_stand_dev.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (x stan. dev.):", self.x_stand_dev)

        self.y_mean = qt.QLabel()
        self.y_mean.setText("0")
        self.y_mean.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (y mean):", self.y_mean)

        self.y_stand_dev = qt.QLabel()
        self.y_stand_dev.setText("0")
        self.y_stand_dev.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (y stan. dev.):", self.y_stand_dev)

        self.amp = qt.QLabel()
        self.amp.setText("0")
        self.amp.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (amp.):", self.amp)

        self.offset = qt.QLabel()
        self.offset.setText("0")
        self.offset.setStyleSheet("background-color: gray;")
        indicator_frame.addRow("2D gaussian fit (offset):", self.offset)

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

    def set_expo_time(self, time, unit):
        t = self.parent.device.set_expo_time(time, unit)
        self.expo_le.setText(t)

    def load_settings(self):
        pass

    def save_settings(self):
        pass


class Plots(qt.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent


class CameraGUI(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('pco.pixelfly usb')
        self.setStyleSheet("QWidget{font: 10pt;}")

        self.device = pixelfly(self)
        self.controls = Controls(self)
        self.plots = Plots(self)

        self.splitter = qt.QSplitter()
        self.splitter.setOrientation(PyQt5.QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        self.splitter.addWidget(self.plots)
        self.splitter.addWidget(self.controls)

        self.resize(1600, 900)
        self.show()


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window = CameraGUI()
    app.exec_()
    main_window.device.cam.close()
    sys.exit(0)
