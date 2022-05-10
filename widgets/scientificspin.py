# Adapted from https://gist.github.com/jdreaver/0be2e44981159d0854f5

# Regular expression to find floats. Match groups are the whole string, the
# whole coefficient, the decimal part of the coefficient, and the exponent
# part.

import re
import numpy as np
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
from widgets.NewWidgets import NewDoubleSpinBox

_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')

_float_re_2 = re.compile(r'(([+-]?\d*(\.\d*)?)([eE][+-]?\d*)?)')

def valid_float_string(string):
    """Used to check if a string represents a valid number"""

    match = _float_re.search(string)
    return match.groups()[0] == string if match else False

def valid_float_string_2(string):
    """Used to check if a string represents a valid number being modified (a numebr missing some parts)"""
    match = _float_re_2.search(string)
    return match.groups()[0] == string if match else False

class FloatValidator(QtGui.QValidator):

    def validate(self, string, position):
        if valid_float_string(string):
            return self.Acceptable, string, position
            
        # if string == "" or string[position-1] in 'eE.-+' or string[0] in 'eE':
        #     return self.Intermediate, string, position

        if valid_float_string_2(string):
            return self.Intermediate, string, position

        return self.Invalid, string, position

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""


class ScientificDoubleSpinBox(NewDoubleSpinBox):

    def __init__(self, range=None, decimals=2, suffix=None):

        # for some reason, I need to put the following two lines before super().__init__()
        self.decimals = decimals
        self.validator = FloatValidator()

        super().__init__(range=range, suffix=suffix)
        self.setDecimals(100)

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        return format_float(self.decimals, value)

    def stepBy(self, steps):
        # Adpated from https://stackoverflow.com/questions/71137584/change-singlestep-in-a-qdoublespinbox-depending-on-the-cursor-position-when-usin

        cursor_position = self.lineEdit().cursorPosition()
        prefix_len = len(self.prefix())
        text = self.cleanText()
        text_len = len(text)
        if cursor_position > prefix_len + text_len:
            cursor_position = prefix_len + text_len
        cursor_position -= prefix_len

        try:
            text_coefficient, text_exp = text.split("e") # text should be in form of "1.23e+3"
        except Exception:
            return

        if cursor_position <= len(text_coefficient):
            # if cursor is to change the coefficient part of the number
            text_int = text_coefficient.split(".")[0] # get the integer part of the text

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

            # perform the step
            value = float(text_coefficient)
            value += steps*single_step
            value *= 10**float(text_exp)
            
            # number of digits won't change so there's no need to handle the case where cursor position should change
        else:
            # if cursor is to change the exponent of the number
            # the first character in text_exp should be + or - sign
            cursor_position -= len(text_coefficient)
            cursor_position -= 1 # count from the right of 'e'
            n_chars = len(text_exp)
            if cursor_position <= 1:
                single_step = 10 ** (n_chars - 2)
            else:
                # if cursor is on the right of the first decimal place
                single_step = 10 ** (n_chars - cursor_position)

            # perform the step
            value = float(text_exp)
            value += steps*single_step
            value = float(text_coefficient)*(10**value)

        self.setValue(value)

        # Undo selection of the whole text.
        self.lineEdit().deselect()


def format_float(decimals, value):
    """Modified form of the 'g' format specifier."""

    # string = ("{:." + f"{decimals}" + "g}").format(value).replace("e+", "e")
    string = np.format_float_scientific(value, precision=decimals, unique=False, exp_digits=1)
    # string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string