import re
import sys
import h5py
import time
import logging
import traceback
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import configparser
import PyQt5
import pyqtgraph as pg
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
import os
import pco
import qdarkstyle # see https://github.com/ColinDuquesnoy/QDarkStyleSheet


# steal colormap data from matplotlib
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
    gauss = np.exp(-dst)*amp + np.random.random_sample(size=(x_range, y_range))/10*amp
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
    signal = PyQt5.QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.image_type = ["background", "signal"]
        self.counter = 0
        self.camera_count_list = []
        self.img_ave = np.zeros((self.parent.device.img_range["xmax"]-self.parent.device.img_range["xmin"]+1,
                                    self.parent.device.img_range["ymax"]-self.parent.device.img_range["ymin"]+1))
        self.parent.device.cam.record(number_of_images=4, mode='ring buffer')
        # number_of_images is buffer size in ring buffer mode, and has to be at least 4
        self.last_time = time.time()

    def run(self):
        while len(self.camera_count_list) < self.parent.device.num_img_to_take and self.parent.control.active:
            type = self.image_type[self.counter%2] # odd-numbered image is background, even-numbered image is signal
            self.counter += 1

            if self.parent.device.trig_mode == "auto sequence":
                pass
            elif self.parent.device.trig_mode == "software trigger":
                self.parent.device.cam.sdk.force_trigger() # softwarely trigger pixelfly camera
            elif self.parent.device.trig_mode =="external/software trigger":
                if self.parent.device.trigger_source == "software":
                    self.parent.device.cam.sdk.force_trigger()
            else:
                print("Trigger mode not supported.")

            while True and self.parent.control.active:
                if self.parent.device.cam.rec.get_status()['dwProcImgCount'] >=self.counter:
                    print(self.parent.device.cam.rec.get_status()['dwProcImgCount'])
                    break
                time.sleep(0.001)

            if self.parent.control.active:
                image, meta = self.parent.device.cam.image(image_number=0xFFFFFFFF) # readout the lastest image
                # image is in "unit16" data type
                image = np.flip(image.T, 1)
                image = image[self.parent.device.img_range["xmin"] : self.parent.device.img_range["xmax"]+1,
                            self.parent.device.img_range["ymin"] : self.parent.device.img_range["ymax"]+1]

                if type == "background":
                    self.image_bg = image
                    self.img_dict = {}
                    self.img_dict["type"] = "background"
                    self.img_dict["image"] = image
                    self.signal.emit(self.img_dict)

                elif type == "signal":
                    cc = np.sum(image-self.image_bg)
                    self.camera_count_list.append(cc)
                    num = len(self.camera_count_list)
                    self.img_dict = {}
                    self.img_dict["type"] = "signal"
                    self.img_dict["num_image"] = num
                    self.img_dict["image"] = image
                    self.img_dict["image_bgsub"] = image.astype("float64") - self.image_bg.astype("float64") # "uint16" can't represent negative number
                    self.img_ave = np.average(np.array([self.img_ave, self.img_dict["image_bgsub"]]), axis=0, weights=[(num-1)/num, 1/num])
                    self.img_dict["image_ave"] = self.img_ave
                    self.img_dict["camera_count"] = np.format_float_scientific(cc, precision=4)
                    self.img_dict["camera_count_ave"] = np.format_float_scientific(np.mean(self.camera_count_list), precision=4)
                    self.img_dict["camera_count_err"] = np.format_float_scientific(np.std(self.camera_count_list)/np.sqrt(num), precision=4)
                    self.signal.emit(self.img_dict)

                else:
                    print("Image type not supported.")

                # Not sure about the reason, but if I just update imges in the main thread from here, it sometimes work but sometimes not.
                # It seems that such signal-slot way is preferred,
                # e.g. https://stackoverflow.com/questions/54961905/real-time-plotting-using-pyqtgraph-and-threading

                print(f"{self.counter}: "+"{:.5f}".format(time.time()-self.last_time))

        self.parent.device.cam.stop()


class pixelfly:
    def __init__(self, parent):
        self.parent = parent

        try:
            self.cam = pco.Camera(interface='USB 2.0')
        except Exception as err:
            logging.error(traceback.format_exc())
            logging.error("Can't open camera")
            return

        self.num_img_to_take = self.parent.defaults["image_number"].getint("default")
        self.img_range = {"xmin": self.parent.defaults["image_range_xmin"].getint("default"),
                          "xmax": self.parent.defaults["image_range_xmax"].getint("default"),
                          "ymin": self.parent.defaults["image_range_ymin"].getint("default"),
                          "ymax": self.parent.defaults["image_range_ymax"].getint("default")}
        self.gaussian_fit = self.parent.defaults["gaussian_fit"].getboolean("default")
        self.img_save = self.parent.defaults["gaussian_fit"].getboolean("default")

        self.set_sensor_format(self.parent.defaults["sensor_format"]["default"])
        self.set_clock_rate(self.parent.defaults["clock_rate"]["default"])
        self.set_conv_factor(self.parent.defaults["conv_factor"]["default"])
        self.set_trigger_mode(self.parent.defaults["trigger_mode"]["default"])
        self.trigger_source = self.parent.defaults["trigger_source"]["default"]
        self.set_expo_time(self.parent.defaults["expo_time"].getfloat("default"))
        self.set_binning(self.parent.defaults["binning"].getint("horizontal_default"),
                        self.parent.defaults["binning"].getint("vertical_default"))

    def set_num_img(self, val):
        self.num_img_to_take = val
        # print(f"number of images to take = {val}")

    def set_img_range(self, text, val):
        self.img_range[text] = val
        # print(f"image range {text} = {val}")

    def set_gauss_fit(self, state):
        self.gaussian_fit = state
        # print(f"2D Gaussian fit = {state}")

    def set_img_save(self, state):
        self.img_save = state
        # print(f"image auto save = {state}")

    def set_sensor_format(self, arg):
        format = self.parent.defaults["sensor_format"][arg]
        self.cam.sdk.set_sensor_format(format)
        self.cam.sdk.arm_camera()
        # print(f"sensor format = {arg}")

    def set_clock_rate(self, arg):
        rate = self.parent.defaults["clock_rate"].getint(arg)
        self.cam.configuration = {"pixel rate": rate}
        # print(f"clock rate = {arg}")

    def set_conv_factor(self, arg):
        conv = self.parent.defaults["conv_factor"].getint(arg)
        self.cam.sdk.set_conversion_factor(conv)
        self.cam.sdk.arm_camera()
        # print(f"conversion factor = {arg}")

    def set_trigger_mode(self, arg):
        self.trig_mode = arg
        mode = self.parent.defaults["trigger_mode"][arg]
        self.cam.configuration = {"trigger": mode}
        # print(f"trigger mode = {arg}")

    def set_trigger_source(self, text, checked):
        if checked:
            self.trigger_source = text
            # print(f"trigger source = {arg}")

    def set_expo_time(self, expo_time):
        self.cam.configuration = {'exposure time': expo_time}
        # print(f"exposure time (in seconds) = {expo_time}")

    def set_binning(self, bin_h, bin_v):
        self.cam.configuration = {'binning': (int(bin_h), int(bin_v))}
        # print(f"binning = {bin_h} (horizontal), {bin_v} (vertical)")


class Control(Scrollarea):
    def __init__(self, parent):
        super().__init__(parent, label="", type="vbox")
        self.setMaximumWidth(400)
        self.frame.setContentsMargins(0,0,0,0)

        self.place_recording()
        self.place_image_control()
        self.place_cam_control()
        self.place_save_load()

    def place_recording(self):
        record_box = qt.QGroupBox("Recording")
        record_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        record_box.setMaximumHeight(270)
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
        self.camera_count.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.camera_count.setToolTip("after background subtraction")
        record_frame.addWidget(self.camera_count, 1, 1, 1, 2)

        record_frame.addWidget(qt.QLabel("Cam. mean:"), 2, 0, 1, 1)
        self.camera_count_mean = qt.QLabel()
        self.camera_count_mean.setText("0")
        self.camera_count_mean.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.camera_count_mean.setToolTip("after background subtraction")
        record_frame.addWidget(self.camera_count_mean, 2, 1, 1, 2)

        record_frame.addWidget(qt.QLabel("Cam. error:"), 3, 0, 1, 1)
        self.camera_count_err_mean = qt.QLabel()
        self.camera_count_err_mean.setText("0")
        self.camera_count_err_mean.setStyleSheet("QLabel{background-color: gray; font: 20pt}")
        self.camera_count_err_mean.setToolTip("after background subtraction")
        record_frame.addWidget(self.camera_count_err_mean, 3, 1, 1, 2)

    def place_image_control(self):
        img_ctrl_box = qt.QGroupBox("Image Control")
        img_ctrl_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        img_ctrl_frame = qt.QFormLayout()
        img_ctrl_box.setLayout(img_ctrl_frame)
        self.frame.addWidget(img_ctrl_box)

        self.num_img_to_take = qt.QSpinBox()
        num_img_upperlimit = self.parent.defaults["image_number"].getint("max")
        self.num_img_to_take.setRange(1, num_img_upperlimit)
        self.num_img_to_take.setValue(self.parent.defaults["image_number"].getint("default"))
        self.num_img_to_take.valueChanged[int].connect(lambda val: self.parent.device.set_num_img(val))
        img_ctrl_frame.addRow("Num of image to take:", self.num_img_to_take)

        format = self.parent.defaults["sensor_format"]["default"]
        format = self.parent.defaults["sensor_format"][format]
        format_str = format + "_absolute_range"
        x_min = self.parent.defaults[format_str].getint("xmin")
        x_max = self.parent.defaults[format_str].getint("xmax")
        self.x_min_sb = qt.QSpinBox()
        self.x_min_sb.setRange(x_min, x_max-1)
        image_min = self.parent.defaults["image_range_xmin"].getint("default")
        self.x_min_sb.setValue(image_min)
        self.x_max_sb = qt.QSpinBox()
        self.x_max_sb.setRange(x_min+1, x_max)
        image_max = self.parent.defaults["image_range_xmax"].getint("default")
        self.x_max_sb.setValue(image_max)
        self.x_min_sb.valueChanged[int].connect(lambda val, text='xmin', sb=self.x_max_sb:
                                                self.set_img_range(text, val, sb))
        self.x_max_sb.valueChanged[int].connect(lambda val, text='xmax', sb=self.x_min_sb:
                                                self.set_img_range(text, val, sb))

        x_range_box = qt.QWidget()
        x_range_layout = qt.QHBoxLayout()
        x_range_layout.setContentsMargins(0,0,0,0)
        x_range_box.setLayout(x_range_layout)
        x_range_layout.addWidget(self.x_min_sb)
        x_range_layout.addWidget(self.x_max_sb)
        img_ctrl_frame.addRow("X range:", x_range_box)

        y_min = self.parent.defaults[format_str].getint("ymin")
        y_max = self.parent.defaults[format_str].getint("ymax")
        self.y_min_sb = qt.QSpinBox()
        self.y_min_sb.setRange(y_min, y_max-1)
        image_min = self.parent.defaults["image_range_ymin"].getint("default")
        self.y_min_sb.setValue(image_min)
        self.y_max_sb = qt.QSpinBox()
        self.y_max_sb.setRange(y_min+1, y_max)
        image_max = self.parent.defaults["image_range_ymax"].getint("default")
        self.y_max_sb.setValue(image_max)
        self.y_min_sb.valueChanged[int].connect(lambda val, text='ymin', sb=self.y_max_sb:
                                                self.set_img_range(text, val, sb))
        self.y_max_sb.valueChanged[int].connect(lambda val, text='ymax', sb=self.y_min_sb:
                                                self.set_img_range(text, val, sb))

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

        # self.image_width = qt.QLabel()
        # self.image_width.setText("0")
        # self.image_width.setStyleSheet("background-color: gray;")
        # self.image_height = qt.QLabel()
        # self.image_height.setText("0")
        # self.image_height.setStyleSheet("background-color: gray;")
        # image_size_box = qt.QWidget()
        # image_size_layout = qt.QHBoxLayout()
        # image_size_layout.setContentsMargins(0,0,0,0)
        # image_size_box.setLayout(image_size_layout)
        # image_size_layout.addWidget(self.image_width)
        # image_size_layout.addWidget(self.image_height)
        # img_ctrl_frame.addRow("Image width x height:", image_size_box)

        self.gauss_fit_chb = qt.QCheckBox()
        self.gauss_fit_chb.setTristate(False)
        gaus = self.parent.defaults["gaussian_fit"].getboolean("default")
        self.gauss_fit_chb.setChecked(gaus)
        self.gauss_fit_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.gauss_fit_chb.stateChanged[int].connect(lambda state: self.parent.device.set_gauss_fit(state))
        img_ctrl_frame.addRow("2D gaussian fit:", self.gauss_fit_chb)

        self.img_save_chb = qt.QCheckBox()
        self.img_save_chb.setTristate(False)
        save = self.parent.defaults["image_save"].getboolean("default")
        self.img_save_chb.setChecked(save)
        self.img_save_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        self.img_save_chb.stateChanged[int].connect(lambda state: self.parent.device.set_img_save(state))
        img_ctrl_frame.addRow("Image auto save:", self.img_save_chb)

        img_ctrl_frame.addRow("------------------", qt.QWidget())

        self.x_mean = qt.QLabel()
        self.x_mean.setText("0")
        self.x_mean.setStyleSheet("background-color: gray;")
        self.x_stand_dev = qt.QLabel()
        self.x_stand_dev.setText("0")
        self.x_stand_dev.setStyleSheet("background-color: gray;")
        gauss_x_box = qt.QWidget()
        gauss_x_layout = qt.QHBoxLayout()
        gauss_x_layout.setContentsMargins(0,0,0,0)
        gauss_x_box.setLayout(gauss_x_layout)
        gauss_x_layout.addWidget(self.x_mean)
        gauss_x_layout.addWidget(self.x_stand_dev)
        img_ctrl_frame.addRow("2D gaussian fit (x):", gauss_x_box)

        self.y_mean = qt.QLabel()
        self.y_mean.setText("0")
        self.y_mean.setStyleSheet("background-color: gray;")
        self.y_stand_dev = qt.QLabel()
        self.y_stand_dev.setText("0")
        self.y_stand_dev.setStyleSheet("background-color: gray;")
        gauss_y_box = qt.QWidget()
        gauss_y_layout = qt.QHBoxLayout()
        gauss_y_layout.setContentsMargins(0,0,0,0)
        gauss_y_box.setLayout(gauss_y_layout)
        gauss_y_layout.addWidget(self.y_mean)
        gauss_y_layout.addWidget(self.y_stand_dev)
        img_ctrl_frame.addRow("2D gaussian fit (y):", gauss_y_box)

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
        op = [x.strip() for x in self.parent.defaults["sensor_format"]["options"].split(',')]
        for i in op:
            self.sensor_format_cb.addItem(i)
        default = self.parent.defaults["sensor_format"]["default"]
        self.sensor_format_cb.setCurrentText(default)
        self.sensor_format_cb.currentTextChanged[str].connect(lambda val: self.set_sensor_format(val))
        cam_ctrl_frame.addRow("Sensor format:", self.sensor_format_cb)

        self.clock_rate_cb = qt.QComboBox()
        self.clock_rate_cb.setMaximumWidth(200)
        self.clock_rate_cb.setMaximumHeight(20)
        op = [x.strip() for x in self.parent.defaults["clock_rate"]["options"].split(',')]
        for i in op:
            self.clock_rate_cb.addItem(i)
        default = self.parent.defaults["clock_rate"]["default"]
        self.clock_rate_cb.setCurrentText(default)
        self.clock_rate_cb.currentTextChanged[str].connect(lambda val: self.parent.device.set_clock_rate(val))
        cam_ctrl_frame.addRow("Clock rate:", self.clock_rate_cb)

        self.conv_factor_cb = qt.QComboBox()
        self.conv_factor_cb.setMaximumWidth(200)
        self.conv_factor_cb.setMaximumHeight(20)
        self.conv_factor_cb.setToolTip("1/gain, or electrons/count")
        op = [x.strip() for x in self.parent.defaults["conv_factor"]["options"].split(',')]
        for i in op:
            self.conv_factor_cb.addItem(i)
        default = self.parent.defaults["conv_factor"]["default"]
        self.conv_factor_cb.setCurrentText(default)
        self.conv_factor_cb.currentTextChanged[str].connect(lambda val: self.parent.device.set_conv_factor(val))
        cam_ctrl_frame.addRow("Conversion factor:", self.conv_factor_cb)

        self.trigger_mode_cb = qt.QComboBox()
        self.trigger_mode_cb.setMaximumWidth(200)
        self.trigger_mode_cb.setMaximumHeight(20)
        op = [x.strip() for x in self.parent.defaults["trigger_mode"]["options"].split(',')]
        for i in op:
            self.trigger_mode_cb.addItem(i)
        default = self.parent.defaults["trigger_mode"]["default"]
        self.trigger_mode_cb.setCurrentText(default)
        self.trigger_mode_cb.currentTextChanged[str].connect(lambda val: self.set_trigger_mode(val))
        cam_ctrl_frame.addRow("Trigger mode:", self.trigger_mode_cb)

        self.trig_source_rblist = []
        trig_bg = qt.QButtonGroup(self.parent)
        self.trig_box = qt.QWidget()
        self.trig_box.setMaximumWidth(200)
        trig_layout = qt.QHBoxLayout()
        trig_layout.setContentsMargins(0,0,0,0)
        self.trig_box.setLayout(trig_layout)
        op = [x.strip() for x in self.parent.defaults["trigger_source"]["options"].split(',')]
        default = self.parent.defaults["trigger_source"]["default"]
        for i in op:
            trig_source_rb = qt.QRadioButton(i)
            trig_source_rb.setChecked(True if default == i else False)
            trig_source_rb.toggled[bool].connect(lambda val, rb=trig_source_rb: self.parent.device.set_trigger_source(rb.text(), val))
            self.trig_source_rblist.append(trig_source_rb)
            trig_bg.addButton(trig_source_rb)
            trig_layout.addWidget(trig_source_rb)
        cam_ctrl_frame.addRow("Trigger source:", self.trig_box)
        self.trig_box.setEnabled(self.parent.defaults["trigger_source"].getboolean("enable"))

        self.expo_le = qt.QLineEdit() # try qt.QDoubleSpinBox() ?
        default = self.parent.defaults["expo_time"].getfloat("default")
        default_unit = self.parent.defaults["expo_unit"]["default"]
        default_unit_num = self.parent.defaults["expo_unit"].getfloat(default_unit)
        default_time = str(default/default_unit_num)
        self.expo_le.setText(default_time)
        self.expo_unit_cb = qt.QComboBox()
        self.expo_unit_cb.setMaximumHeight(30)
        op = [x.strip() for x in self.parent.defaults["expo_unit"]["options"].split(',')]
        for i in op:
            self.expo_unit_cb.addItem(i)
        self.expo_unit_cb.setCurrentText(default_unit)
        self.expo_le.editingFinished.connect(lambda le=self.expo_le, cb=self.expo_unit_cb:
                                            self.set_expo_time(le.text(), cb.currentText()))
        self.expo_unit_cb.currentTextChanged[str].connect(lambda val, le=self.expo_le: self.set_expo_time(le.text(), val))
        expo_box = qt.QWidget()
        expo_box.setMaximumWidth(200)
        expo_layout = qt.QHBoxLayout()
        expo_layout.setContentsMargins(0,0,0,0)
        expo_box.setLayout(expo_layout)
        expo_layout.addWidget(self.expo_le)
        expo_layout.addWidget(self.expo_unit_cb)
        cam_ctrl_frame.addRow("Exposure time:", expo_box)

        self.bin_hori_cb = qt.QComboBox()
        self.bin_vert_cb = qt.QComboBox()
        op = [x.strip() for x in self.parent.defaults["binning"]["options"].split(',')]
        for i in op:
            self.bin_hori_cb.addItem(i)
            self.bin_vert_cb.addItem(i)
        bin_h = self.parent.defaults["binning"]["horizontal_default"]
        self.bin_hori_cb.setCurrentText(bin_h)
        bin_v = self.parent.defaults["binning"]["vertical_default"]
        self.bin_vert_cb.setCurrentText(bin_v)
        self.bin_hori_cb.currentTextChanged[str].connect(lambda val, text="hori", cb=self.bin_vert_cb: self.set_binning(text, val, cb.currentText()))
        self.bin_vert_cb.currentTextChanged[str].connect(lambda val, text="vert", cb=self.bin_hori_cb: self.set_binning(text, cb.currentText(), val))
        bin_box = qt.QWidget()
        bin_box.setMaximumWidth(200)
        bin_layout = qt.QHBoxLayout()
        bin_layout.setContentsMargins(0,0,0,0)
        bin_box.setLayout(bin_layout)
        bin_layout.addWidget(self.bin_hori_cb)
        bin_layout.addWidget(self.bin_vert_cb)
        cam_ctrl_frame.addRow("Binning H x V:", bin_box)

    def place_save_load(self):
        self.save_load_box = qt.QGroupBox("Save/Load Settings")
        self.save_load_box.setStyleSheet("QGroupBox {border: 1px solid #304249;}")
        save_load_frame = qt.QFormLayout()
        self.save_load_box.setLayout(save_load_frame)
        self.frame.addWidget(self.save_load_box)

        self.file_name_le = qt.QLineEdit()
        default_file_name = self.parent.defaults["file_name_save"]["default"]
        self.file_name_le.setText(default_file_name)
        save_load_frame.addRow("File name to save:", self.file_name_le)

        self.date_time_chb = qt.QCheckBox()
        self.date_time_chb.setTristate(False)
        date = self.parent.defaults["append_time"].getboolean("default")
        self.date_time_chb.setChecked(date)
        self.date_time_chb.setStyleSheet("QCheckBox::indicator {width: 15px; height: 15px;}")
        save_load_frame.addRow("Auto append time:", self.date_time_chb)

        self.save_settings_bt = qt.QPushButton("save settings")
        self.save_settings_bt.setMaximumWidth(200)
        self.save_settings_bt.clicked[bool].connect(lambda val: self.save_settings())
        save_load_frame.addRow("Save settings:", self.save_settings_bt)

        self.load_settings_bt = qt.QPushButton("load settings")
        self.load_settings_bt.setMaximumWidth(200)
        self.load_settings_bt.clicked[bool].connect(lambda val: self.load_settings())
        save_load_frame.addRow("Load settings:", self.load_settings_bt)

    def record(self):
        self.active = True

        self.enable_widgets(False)
        self.num_image.setText("0")

        self.rec = CamThread(self.parent)
        self.rec.signal.connect(self.img_ctrl_update)
        self.rec.finished.connect(self.stop)
        self.rec.start()

        # Another way to do this is to use QTimer() to trigger imgae image readout,
        # but in that case, the while loop that waits for the image is running in the main thread,
        # so it will block the main thread.

    def scan(self):
        self.active = True
        self.enable_widgets(False)

    def stop(self):
        if self.active:
            self.active = False
            self.enable_widgets(True)

    @PyQt5.QtCore.pyqtSlot(dict)
    def img_ctrl_update(self, img_dict):
        if img_dict["type"] == "background":
            img = img_dict["image"]
            self.parent.image_win.imgs_dict["Background"].setImage(img)

        elif img_dict["type"] == "signal":
            img = img_dict["image"]
            self.parent.image_win.imgs_dict["Signal"].setImage(img)
            img = img_dict["image_bgsub"]
            self.parent.image_win.imgs_dict["Signal w/ bg subtraction"].setImage(img)
            self.parent.image_win.x_plot.setData(np.sum(img, axis=1))
            self.parent.image_win.y_plot.setData(np.sum(img, axis=0))
            self.parent.image_win.ave_img.setImage(img_dict["image_ave"])

            self.num_image.setText(str(img_dict["num_image"]))
            self.camera_count.setText(str(img_dict["camera_count"]))
            self.camera_count_mean.setText(str(img_dict["camera_count_ave"]))
            self.camera_count_err_mean.setText(str(img_dict["camera_count_err"]))

    def enable_widgets(self, arg):
        self.stop_bt.setEnabled(not arg)
        self.record_bt.setEnabled(arg)
        self.scan_bt.setEnabled(arg)
        self.num_img_to_take.setEnabled(arg)
        self.x_min_sb.setEnabled(arg)
        self.x_max_sb.setEnabled(arg)
        self.y_min_sb.setEnabled(arg)
        self.y_max_sb.setEnabled(arg)
        self.gauss_fit_chb.setEnabled(arg)
        self.img_save_chb.setEnabled(arg)
        self.cam_ctrl_box.setEnabled(arg)
        self.save_load_box.setEnabled(arg)

        self.parent.app.processEvents()

    def set_img_range(self, text, val, sb):
        if text == "xmin":
            sb.setMinimum(val+1)
        elif text == "xmax":
            sb.setMaximum(val-1)
        elif text == "ymin":
            sb.setMinimum(val+1)
        elif text == "ymax":
            sb.setMaximum(val-1)

        self.parent.device.set_img_range(text, val)

    def set_sensor_format(self, val):
        format = self.parent.defaults["sensor_format"][val]
        format_str = format + "_absolute_range"
        x_max = (self.parent.defaults[format_str].getint("xmax")+1)/int(self.bin_hori_cb.currentText()) - 1
        self.x_max_sb.setMaximum(int(x_max))
        y_max = (self.parent.defaults[format_str].getint("ymax")+1)/int(self.bin_vert_cb.currentText()) - 1
        self.y_max_sb.setMaximum(int(y_max))
        # number in both 'min' and 'max' spinboxes will adjusted automatically
        self.parent.device.set_sensor_format(val)

    def set_trigger_mode(self, val):
        if val == "external/software trigger":
            self.trig_box.setEnabled(True)
        else:
            self.trig_box.setEnabled(False)

        self.parent.device.set_trigger_mode(val)

    def set_expo_time(self, time, unit):
        unit_num = self.parent.defaults["expo_unit"].getfloat(unit)
        try:
            expo_time = float(time)*unit_num
        except ValueError as err:
            logging.warning(traceback.format_exc())
            logging.warning("Exposure time invalid!")
            return

        expo_decimals = self.parent.defaults["expo_time"].getint("decimals")
        expo_min = self.parent.defaults["expo_time"].getfloat("min")
        expo_max = self.parent.defaults["expo_time"].getfloat("max")
        expo_time_round = round(expo_time, expo_decimals)
        if expo_time_round < expo_min:
            expo_time_round = expo_min
        elif expo_time_round > expo_max:
            expo_time_round = expo_max

        d = int(expo_decimals+np.log10(unit_num))
        if d:
            t = round(expo_time_round/unit_num, d)
            t = f"{t}"
        else:
            t = "{:d}".format(round(expo_time_round/unit_num))

        self.expo_le.setText(t)
        self.parent.device.set_expo_time(expo_time_round)

    def set_binning(self, text, bin_h, bin_v):
        format = self.parent.defaults["sensor_format"][self.sensor_format_cb.currentText()]
        format_str = format + "_absolute_range"
        if text == "hori":
            x_max = (self.parent.defaults[format_str].getint("xmax")+1)/int(bin_h) - 1
            self.x_max_sb.setMaximum(int(x_max))
        elif text == "vert":
            y_max = (self.parent.defaults[format_str].getint("ymax")+1)/int(bin_v) - 1
            self.y_max_sb.setMaximum(int(y_max))
        else:
            print("Binning type not supported.")

        self.parent.device.set_binning(bin_h, bin_v)

    def save_settings(self):
        file_name = ""
        if self.file_name_le.text():
            file_name += self.file_name_le.text()
        if self.date_time_chb.isChecked():
            if file_name != "":
                file_name += "_"
            file_name += time.strftime("%Y%m%d_%H%M%S")
        file_name += ".ini"
        file_name = r"saved_settings/"+file_name
        if os.path.exists(file_name):
            overwrite = qt.QMessageBox.warning(self, 'File name exists',
                                            'File name already exists. Continue to overwrite it?',
                                            qt.QMessageBox.Yes | qt.QMessageBox.No,
                                            qt.QMessageBox.No)
            if overwrite == qt.QMessageBox.No:
                return

        config = configparser.ConfigParser()

        config["image_control"] = {}
        config["image_control"]["num_image"] = str(self.num_img_to_take.value())
        config["image_control"]["xmin"] = str(self.x_min_sb.value())
        config["image_control"]["xmax"] = str(self.x_max_sb.value())
        config["image_control"]["ymin"] = str(self.y_min_sb.value())
        config["image_control"]["ymax"] = str(self.y_max_sb.value())
        config["image_control"]["2D_gaussian_fit"] = str(self.gauss_fit_chb.isChecked())
        config["image_control"]["image_auto_save"] = str(self.img_save_chb.isChecked())

        config["camera_control"] = {}
        config["camera_control"]["sensor_format"] = self.sensor_format_cb.currentText()
        config["camera_control"]["clock_rate"] = self.clock_rate_cb.currentText()
        config["camera_control"]["conversion_factor"] = self.conv_factor_cb.currentText()
        config["camera_control"]["trigger_mode"] = self.trigger_mode_cb.currentText()
        for i in self.trig_source_rblist:
            if i.isChecked():
                t = i.text()
                break
        config["camera_control"]["trigger_source"] = t
        config["camera_control"]["exposure_time"] = self.expo_le.text()
        config["camera_control"]["exposure_unit"] = self.expo_unit_cb.currentText()
        config["camera_control"]["binning_horizontal"] = self.bin_hori_cb.currentText()
        config["camera_control"]["binning_vertical"] = self.bin_vert_cb.currentText()

        configfile = open(file_name, "w")
        config.write(configfile)
        configfile.close()

    def load_settings(self):
        file_name, _ = qt.QFileDialog.getOpenFileName(self,"Load settigns", "saved_settings/", "All Files (*);;INI File (*.ini)")
        if not file_name:
            return

        config = configparser.ConfigParser()
        config.read(file_name)

        self.num_img_to_take.setValue(config["image_control"].getint("num_image"))
        # the spinbox emits 'valueChanged' signal, and its connected function will be called
        self.x_min_sb.setValue(config["image_control"].getint("xmin"))
        self.x_max_sb.setValue(config["image_control"].getint("xmax"))
        self.y_min_sb.setValue(config["image_control"].getint("ymin"))
        self.y_max_sb.setValue(config["image_control"].getint("ymax"))
        self.y_max_sb.setValue(config["image_control"].getint("ymax"))
        self.gauss_fit_chb.setChecked(config["image_control"].getboolean("2d_gaussian_fit"))
        # the combobox emits 'stateChanged' signal, and its connected function will be called
        self.img_save_chb.setChecked(config["image_control"].getboolean("image_auto_save"))

        self.sensor_format_cb.setCurrentText(config["camera_control"]["sensor_format"])
        # the combobox emits 'currentTextChanged' signal, and its connected function will be called
        # make sure sensor format is updated after image range settings
        self.clock_rate_cb.setCurrentText(config["camera_control"]["clock_rate"])
        self.conv_factor_cb.setCurrentText(config["camera_control"]["conversion_factor"])
        for i in self.trig_source_rblist:
            if i.text() == config["camera_control"]["trigger_source"]:
                i.setChecked(True)
                break
        self.trigger_mode_cb.setCurrentText(config["camera_control"]["trigger_mode"])
        # make sure trigger mode is updated after trigger source, because for some trigger modes, trigger source box will be disabled
        self.expo_le.setText(config["camera_control"]["exposure_time"])
        # QLineEdit won't emit 'editingfinishede signal
        self.expo_unit_cb.setCurrentText(config["camera_control"]["exposure_unit"])
        # make sure exposure unit is updated after exposure time QLineEdit, so the pixelfly.set_expo_time functioni will be called
        self.bin_hori_cb.setCurrentText(config["camera_control"].get("binning_horizontal"))
        self.bin_vert_cb.setCurrentText(config["camera_control"].get("binning_vertical"))
        # make sure binning is updated after image range settings

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
            self.img_tab.addTab(graphlayout, " "+name+" ")
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

        self.img_tab.setCurrentIndex(2) # make tab #2 (count from 0) to show as default

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

        data = fake_data()
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
        self.setWindowTitle('pco.pixelfly usb (ring buffer)')
        self.setStyleSheet("QWidget{font: 10pt;}")
        # self.setStyleSheet("QToolTip{background-color: black; color: white; font: 10pt;}")
        self.app = app

        self.defaults = configparser.ConfigParser()
        self.defaults.read('defaults.ini')

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
