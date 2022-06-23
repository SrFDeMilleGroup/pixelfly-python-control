import pyqtgraph as pg
import PyQt5
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
import logging
import numpy as np

class NewBox(qt.QGroupBox):
    """A formated QGroupBox with a layout attached."""

    def __init__(self, layout_type="grid"):
        super().__init__()
        # self.setStyleSheet("QGroupBox {border: 0px;}")
        if layout_type == "grid":
            self.frame = qt.QGridLayout()
        elif layout_type == "vbox":
            self.frame = qt.QVBoxLayout()
        elif layout_type == "hbox":
            self.frame = qt.QHBoxLayout()
        elif layout_type == "form":
            self.frame = qt.QFormLayout()
            self.frame.setHorizontalSpacing(0)
            self.setStyleSheet("QGroupBox {border: 0px; padding-left: 0; padding-right: 0;}")
        elif layout_type == "flexgrid":
            self.frame = FlexibleGridLayout()
        else:
            logging.warning("NewBox: layout type not supported.")
            self.frame = qt.QGridLayout()
        self.frame.setContentsMargins(0,0,0,0)
        self.setLayout(self.frame)


class NewDoubleSpinBox(qt.QDoubleSpinBox):
    """
    A doublespinbox that won't respond if the mouse just hovers over it and scrolls the wheel.
    It will respond if it's clicked and get focus.
    """

    def __init__(self, range=None, decimals=None, suffix=None):
        super().__init__()

        # mouse hovering over this widget and scrolling the wheel won't bring focus into it
        # mouse can bring focus to this widget by clicking it
        self.setFocusPolicy(PyQt5.QtCore.Qt.StrongFocus)

        # scroll event and up/down button still emit valuechanged signal,
        # but typing value through keyboard only emits valuecahnged signal when enter is pressed or focus is lost
        self.setKeyboardTracking(False)

        # 0 != None
        # don't use "if not range:" statement, in case range is set to zero
        if range != None:
            self.setRange(range[0], range[1])
        else:
            self.setRange(-np.inf, np.inf)
        if decimals != None:
            self.setDecimals(decimals)
        if suffix != None:
            self.setSuffix(suffix)

    def stepBy(self, steps):
        # Adpated from https://stackoverflow.com/questions/71137584/change-singlestep-in-a-qdoublespinbox-depending-on-the-cursor-position-when-usin

        cursor_position = self.lineEdit().cursorPosition()
        prefix_len = len(self.prefix())
        text = self.cleanText()
        text_len = len(text)
        if cursor_position > prefix_len + text_len:
            cursor_position = prefix_len + text_len
        cursor_position -= prefix_len

        text_int = text.split(".")[0] # get the integer part of the text

        # number of characters before the decimal separator including - sign (+ sign is omitted by default)
        n_chars_before_sep = len(text_int)

        if text_int[0] == '-':
            # if the first character is '-' sign
            if cursor_position <= 1:
                single_step = 10 ** (n_chars_before_sep - 2)
            elif cursor_position <= n_chars_before_sep + 1:
                # if cursor is on the left of the first decimal place
                single_step = 10 ** (n_chars_before_sep - cursor_position)
            else:
                # if cursor is on the right of the first decimal place
                single_step = 10 ** (n_chars_before_sep - cursor_position + 1)
        else:
            if cursor_position <= 0:
                single_step = 10 ** (n_chars_before_sep - 1)
            elif cursor_position <= n_chars_before_sep + 1:
                # if cursor is on the left of the first decimal place
                single_step = 10 ** (n_chars_before_sep - cursor_position)
            else:
                # if cursor is on the right of the first decimal place
                single_step = 10 ** (n_chars_before_sep - cursor_position + 1)

        # Change single step and perform the step
        self.setSingleStep(single_step)
        super().stepBy(steps)

        # Undo selection of the whole text.
        self.lineEdit().deselect()

        # Handle cases where the number of characters before the decimal separator changes. Step size should remain the same.
        text = self.cleanText()
        text_int = text.split(".")[0] # get the integer part of the text
        new_n_chars_before_sep = len(text_int)

        cursor_position += (new_n_chars_before_sep - n_chars_before_sep)
        cursor_position += prefix_len
        self.lineEdit().setCursorPosition(cursor_position)

    def wheelEvent(self, event):
        """Modify the wheelEvent so this widget only responds when it has focus."""
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            # if the event is ignored, it will be passed to and handled by parent widget
            event.ignore()

class NewSpinBox(qt.QSpinBox):
    """
    A spinbox 
    1. that won't respond if the mouse just hovers over it and scrolls the wheel. It will respond if it's clicked and get focus;
    2. whose stepsize depends on cursor position
    """

    def __init__(self, range=None, suffix=None):
        super().__init__()

        # mouse hovering over this widget and scrolling the wheel won't bring focus into it
        # mouse can bring focus to this widget by clicking it
        self.setFocusPolicy(PyQt5.QtCore.Qt.StrongFocus)

        # scroll event and up/down button still emit valuechanged signal,
        # but typing value through keyboard only emits valuecahnged signal when enter is pressed or focus is lost
        self.setKeyboardTracking(False)

        if range != None:
            self.setRange(range[0], range[1])
        else:
            self.setRange(-2147483648, 2147483647) # max range spin box can accept
        if suffix != None:
            self.setSuffix(suffix)

    def stepBy(self, steps):
        # Adpated from https://stackoverflow.com/questions/71137584/change-singlestep-in-a-qdoublespinbox-depending-on-the-cursor-position-when-usin

        cursor_position = self.lineEdit().cursorPosition()
        prefix_len = len(self.prefix())
        text = self.cleanText()
        text_len = len(text)
        if cursor_position > prefix_len + text_len:
            cursor_position = prefix_len + text_len
        cursor_position -= prefix_len

        # number of characters including - sign (+ sign is omitted by default)
        n_chars = len(text)

        if text[0] == '-':
            # if the first character is '-' sign
            if cursor_position == 0 or cursor_position == 1:
                single_step = 10 ** (n_chars - 2)
            else:
                single_step = 10 ** (n_chars - cursor_position)
        else:
            if cursor_position == 0:
                single_step = 10 ** (n_chars - 1)
            else:
                single_step = 10 ** (n_chars - cursor_position)

        # Change single step and perform the step
        self.setSingleStep(single_step)
        super().stepBy(steps)

        # Undo selection of the whole text.
        self.lineEdit().deselect()

        # Handle cases where the number of characters changes. Step size should remain the same.
        new_n_chars = len(self.cleanText())

        cursor_position += (new_n_chars - n_chars)
        cursor_position += prefix_len
        self.lineEdit().setCursorPosition(cursor_position)

    def wheelEvent(self, event):
        """Modify the wheelEvent so this widget only responds when it has focus."""

        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class NewComboBox(qt.QComboBox):
    """
    A combobox that won't respond if the mouse just hovers over it and scrolls the wheel.
    It will respond if it's clicked and get focus.
    """

    def __init__(self, item_list=None, current_item=None):
        super().__init__()

        # mouse hovering over this widget and scrolling the wheel won't bring focus into it
        # mouse can bring focus to this widget by clicking it
        self.setFocusPolicy(PyQt5.QtCore.Qt.StrongFocus)

        if item_list != None:
            self.addItems(item_list)
        if current_item != None:
            self.setCurrentText(current_item)

        # self.setStyleSheet("QComboBox::down-arrow{padding-left:0px;}")
        # self.setStyleSheet("QComboBox {padding:0px;}")

    def wheelEvent(self, event):
        """Modify the wheelEvent so this widget only responds when it has focus."""

        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class NewPlot(pg.PlotWidget):
    """A formated plot widget"""

    def __init__(self, parent=None):
        super().__init__()
        tickstyle = {"showValues": False}

        self.showGrid(True, True)
        self.setLabel("top")
        self.getAxis("top").setStyle(**tickstyle)
        self.setLabel("right")
        self.getAxis("right").setStyle(**tickstyle)

        self.getAxis("bottom").enableAutoSIPrefix(False)

# create a scroll area of a specific layout, e.g. form, grid, vbox, etc
# class scrollArea(qt.QGroupBox):
#     def __init__(self, layout_type="grid"):
#         super().__init__()
#         outer_layout = qt.QGridLayout()
#         outer_layout.setContentsMargins(0,0,0,0)
#         self.setLayout(outer_layout)

#         scroll = qt.QScrollArea()
#         scroll.setStyleSheet("QScrollArea{border-width:0px}")
#         scroll.setWidgetResizable(True)
#         scroll.setFrameStyle(0x10) # see https://doc.qt.io/qt-5/qframe.html for different frame styles
#         outer_layout.addWidget(scroll)

#         box = NewBox(layout_type=layout_type)
#         box.setStyleSheet("QGroupBox{border-width: 0px}")
#         scroll.setWidget(box)

#         self.frame = box.frame

class NewScrollArea(qt.QScrollArea):
    """A scrollarea with a frame attached."""

    def __init__(self, layout_type="grid"):
        super().__init__()

        self.setWidgetResizable(True)
        self.setFrameStyle(0x10) # see https://doc.qt.io/qt-5/qframe.html for different frame styles

        box = NewBox(layout_type=layout_type)
        box.setObjectName("ScrollArea")
        box.setStyleSheet("QGroupBox#ScrollArea{border-width: 0px;}")
        self.setWidget(box)

        self.frame = box.frame

# create a scroll area of a specific layout, e.g. form, grid, vbox, etc
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
        scroll.setFrameStyle(0x10) # see https://doc.qt.io/qt-5/qframe.html for different frame styles
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
            logging.warning("Frame type not supported!")

        box.setLayout(self.frame)

# https://github.com/js216/CeNTREX/blob/master/main.py
class FlexibleGridLayout(qt.QHBoxLayout):
    """A QHBoxLayout of QVBoxLayouts."""
    def __init__(self, grid_num=10):
        super().__init__()
        self.cols = {}

        # populate the grid with placeholders
        for col in range(grid_num):
            self.cols[col] = qt.QVBoxLayout()
            self.addLayout(self.cols[col])

            # add stretchable spacer to prevent stretching the device controls boxes
            self.cols[col].addStretch()

            # reverse the layout order to keep the spacer always at the bottom
            self.cols[col].setDirection(qt.QBoxLayout.BottomToTop)

            # add horizontal placeholders
            vbox = self.cols[col]
            for row in range(grid_num):
                vbox.addLayout(qt.QHBoxLayout())

    def addWidget(self, widget, row, col):
        vbox = self.cols[col]
        rev_row = vbox.count() - 1 - row
        placeholder = vbox.itemAt(rev_row).layout()
        if not placeholder.itemAt(0):
            placeholder.addWidget(widget)

    def clear(self):
        """Remove all widgets."""
        for col_num, col in self.cols.items():
            for i in reversed(range(col.count())):
                try:
                    if col.itemAt(i).layout():
                        col.itemAt(i).layout().itemAt(0).widget().setParent(None)
                except AttributeError:
                    logging.info("Exception in clear() in class FlexibleGridLayout", exc_info=True)
                    pass