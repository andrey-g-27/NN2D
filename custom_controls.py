# Variety of custom Qt-based controls for the main program.

import math as m

import numpy as np
import numexpr as ne

from PyQt6.QtWidgets import \
    QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QLineEdit, QTableWidget, \
    QTableWidgetItem, QSlider, QDoubleSpinBox, QAbstractSpinBox

from PyQt6.QtGui import QDoubleValidator, QIntValidator

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from threaded_figure import \
    PlotHandler, LearningStrengthPlotUpdater, LearningStrengthPlotData

class AutoLimitIntLineEdit(QLineEdit):
    def __init__(self, start_val, min_val, max_val):
        QLineEdit.__init__(self, start_val)
        self.setValidator(QIntValidator(min_val, max_val))
        self._min_val = min_val
        self._max_val = max_val
        self.textEdited.connect(self._proc_text_ed)

    def _proc_text_ed(self, text):
        if ((text == "") or (int(text) < self._min_val)):
            self.setText(str(self._min_val))
        elif (int(text) > self._max_val):
            self.setText(str(self._max_val))

class NNInnerStructureTable(QTableWidget):
    def __init__(self):
        QTableWidget.__init__(self)
        self._default_cell_role = Qt.ItemDataRole.DisplayRole
        self._default_cell_value = ""
        self._item_prototype = QTableWidgetItem()
        self._item_prototype.setData(self._default_cell_role, self._default_cell_value)

        self.setRowCount(2)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(["Layer size"])

        self.setItem(0, 0, self._item_prototype.clone())
        self.item(0, 0).setData(self._default_cell_role, "2")
        self.setItem(1, 0, self._item_prototype.clone())
        
        self.cellChanged.connect(self._cell_changed_reaction)

    @pyqtSlot(int, int)
    def _cell_changed_reaction(self, row, col):
        self.cellChanged.disconnect(self._cell_changed_reaction)
        curr_value = self.item(row, col).data(self._default_cell_role)
        try:
            curr_value = int(curr_value)
        except Exception:
            curr_value = -1
        if (curr_value <= 0):
            self.item(row, col).setData(self._default_cell_role, "")
        self._shrink_column()
        self.cellChanged.connect(self._cell_changed_reaction)
        self.struct_changed.emit()

    def _shrink_column(self):
        # produces error 
        # "QAbstractItemView::closeEditor called with an editor that does not belong to this view"
        # but it is not fatal and seems to not affect functionality
        # so code stays like this for now
        for row in range(self.rowCount() - 1, -1, -1):
            curr_value = self.item(row, 0).data(self._default_cell_role)
            if (curr_value == ""):
                self.removeRow(row)
        self.insertRow(self.rowCount())
        self.setItem(self.rowCount() - 1, 0, self._item_prototype.clone())
    
    def get_layer_sizes(self):
        sizes = []
        for row in range(self.rowCount()):
            curr_size = self.item(row, 0).data(self._default_cell_role)
            if (curr_size != ""):
                sizes.append(int(curr_size))
        return sizes

    struct_changed = pyqtSignal()

class NNLearningStrengthControl(QVBoxLayout):
    def __init__(self, default_fcn):
        QVBoxLayout.__init__(self)
        self._label_learn_str = QLabel("S(1≤i≤1,000,000) = ")
        self._edit_learn_str = QLineEdit(default_fcn)
        self._plot_handler = PlotHandler(LearningStrengthPlotUpdater)
        self._fig_canv = self._plot_handler.get_canvas()
        self.addWidget(self._label_learn_str)
        self.addWidget(self._edit_learn_str)
        self.addWidget(self._fig_canv)

        self._str_fcn_ok = False

        self._edit_learn_str.textChanged.connect(self._fcn_edit_reaction)
        self._reset_plot.connect(self._plot_handler.reset_plot)
        self._show_plot.connect(self._plot_handler.show_plot)

        self._reset_plot.emit()
        self._fcn_edit_reaction(self._edit_learn_str.text())

    def get_strength_function(self):
        return "{0:s}".format(self._edit_learn_str.text())

    def is_str_fcn_ok(self):
        return self._str_fcn_ok

    def set_enable_edit(self, enable):
        self._edit_learn_str.setEnabled(enable)
    
    @pyqtSlot(str)
    def _fcn_edit_reaction(self, fcn_str):
        i = np.logspace(0.0, 6.0, 101, base = 10.0)
        try:
            s = ne.evaluate(fcn_str)
        except Exception:
            self._str_fcn_ok = False
            self._edit_learn_str.setStyleSheet("border: 1.5px solid #FF4040")
            return
        self._str_fcn_ok = True
        self._edit_learn_str.setStyleSheet("border: 1.5px solid #40FF40")
        plot_data = LearningStrengthPlotData(i, s)
        self._show_plot.emit(plot_data)
        self.learn_str_fcn_changed.emit()

    learn_str_fcn_changed = pyqtSignal()
    _reset_plot = pyqtSignal()
    _show_plot = pyqtSignal(LearningStrengthPlotData)

class DoubleSliderSpinbox(QGridLayout):
    def __init__(self, label, start_value, min_value, max_value):
        QGridLayout.__init__(self)

        self._label = QLabel(label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1000)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(30)

        self._spinbox = QDoubleSpinBox()

        self.addWidget(self._label, 0, 0, 1, 1)
        self.addWidget(self._spinbox, 0, 1, 1, 1)
        self.addWidget(self._slider, 1, 0, 1, 2)
        self.setColumnStretch(1, 1)

        self._slider.valueChanged.connect(self._value_changed_from_slider)
        self._spinbox.valueChanged.connect(self._value_changed_from_spinbox)

        self._value = start_value
        self.set_range(min_value, max_value)
        self.set_value(start_value)

    def set_value(self, new_value, from_slider = False, from_spinbox = False):
        self._value = new_value
        self._slider.valueChanged.disconnect(self._value_changed_from_slider)
        self._spinbox.valueChanged.disconnect(self._value_changed_from_spinbox)
        if (not from_slider):
            self._slider.setValue(self._double_to_slider_int(self._value))
        if (not from_spinbox):
            self._spinbox.setValue(self._value)
        self._slider.valueChanged.connect(self._value_changed_from_slider)
        self._spinbox.valueChanged.connect(self._value_changed_from_spinbox)
        self.value_changed_no_arg.emit()
        self.value_changed.emit(self._value)

    def get_value(self):
        return self._value

    @pyqtSlot(float)
    def set_min_value(self, new_min_val):
        self._min_value = new_min_val
        self._update_range()

    @pyqtSlot(float)
    def set_max_value(self, new_max_val):
        self._max_value = new_max_val
        self._update_range()

    @pyqtSlot(float, float)
    def set_range(self, new_min_val, new_max_val):
        self._min_value = new_min_val
        self._max_value = new_max_val
        self._update_range()

    def set_interaction_enabled(self, enabled):
        self._slider.setEnabled(enabled)
        self._spinbox.setEnabled(enabled)

    def _update_range(self):
        if (self._min_value >= self._max_value):
            common_value = (self._min_value + self._max_value) / 2.0
            self._min_value = common_value
            self._max_value = common_value
        
        new_value = self._value
        if (new_value <= self._min_value):
            new_value = self._min_value
        if (new_value >= self._max_value):
            new_value = self._max_value
        
        self._spinbox.setRange(self._min_value, self._max_value)
        decimals = int(m.ceil(max(0.0, \
            -m.log10((self._max_value - self._min_value) / \
            (self._slider.maximum() - self._slider.minimum())))))
        self._spinbox.setDecimals(decimals)
        self._spinbox.setSingleStep(m.pow(10.0, -decimals + 1.0))
        self._spinbox.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)

        self.set_value(new_value)

    @pyqtSlot(int)
    def _value_changed_from_slider(self, new_value_int):
        self.set_value(self._slider_int_to_double(new_value_int), \
            from_slider = True, from_spinbox = False)

    @pyqtSlot(float)
    def _value_changed_from_spinbox(self, new_value):
        self.set_value(new_value, from_slider = False, from_spinbox = True)

    def _slider_int_to_double(self, val_int):
        return ((val_int - self._slider.minimum()) / \
            (self._slider.maximum() - self._slider.minimum()) * \
            (self._max_value - self._min_value) + \
            self._min_value)

    def _double_to_slider_int(self, val_dbl):
        return int(round((val_dbl - self._min_value) / \
            (self._max_value - self._min_value) * \
            (self._slider.maximum() - self._slider.minimum()) + \
            self._slider.minimum()))

    value_changed_no_arg = pyqtSignal()
    value_changed = pyqtSignal(float)

class RangeEdit(QHBoxLayout):
    def __init__(self, var_name, min_value, max_value):
        QHBoxLayout.__init__(self)

        self._label_left = QLabel("{0:s} from ".format(var_name))
        self._edit_range_min = QLineEdit(min_value)
        self._edit_range_min.setValidator(QDoubleValidator())
        self._label_right = QLabel(" to ")
        self._edit_range_max = QLineEdit(max_value)
        self._edit_range_max.setValidator(QDoubleValidator())
        self.addWidget(self._label_left)
        self.addWidget(self._edit_range_min)
        self.addWidget(self._label_right)
        self.addWidget(self._edit_range_max)

        self._min_value = float(min_value)
        self._max_value = float(max_value)

        self._edit_range_min.editingFinished.connect(self._update_min_value)
        self._edit_range_max.editingFinished.connect(self._update_max_value)

    def get_range(self):
        return (self._min_value, self._max_value)

    def set_line_edits_enabled(self, enabled):
        self._edit_range_min.setEnabled(enabled)
        self._edit_range_max.setEnabled(enabled)

    def _update_min_value(self):
        self._min_value = float(self._edit_range_min.text())
        self.range_changed_no_arg.emit()
        self.range_changed.emit(self._min_value, self._max_value)

    def _update_max_value(self):
        self._max_value = float(self._edit_range_max.text())
        self.range_changed_no_arg.emit()
        self.range_changed.emit(self._min_value, self._max_value)

    range_changed_no_arg = pyqtSignal()
    range_changed = pyqtSignal(float, float)
