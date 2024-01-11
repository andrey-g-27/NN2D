# Program for experimenting with learning 2D mathematical functions with 
# neural network. Provides variety of visualizations and controls.

# Required libraries (for program or its custom required modules): 
#     External: PyQt6, NumPy, Numexpr, Matplotlib.
#     Internal: neural_network, nn_displayer, threaded_figure, custom_controls.

import numpy as np
import numpy.random as rand
import numexpr as ne

import sys

from PyQt6.QtWidgets import \
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, \
    QGroupBox, QPlainTextEdit, QLineEdit, QPushButton, QCheckBox, QButtonGroup, \
    QRadioButton, QStackedWidget, QAbstractButton, QSizePolicy

from PyQt6.QtCore import QObject, QThread, QMutex, pyqtSignal, pyqtSlot

from neural_network import NeuralNetwork
from threaded_figure import \
    ConvPlotData, ContourPlotData, ConvPlotUpdater, ContourPlotUpdater, PlotHandler
from nn_displayer import NNDisplayer
from custom_controls import \
    AutoLimitIntLineEdit, NNInnerStructureTable, NNLearningStrengthControl, \
    DoubleSliderSpinbox, RangeEdit

class MutWrap(object):
    def __init__(self, value):
        self.value = value

class NNState:
    def __init__(self, curr_iter, conv_val, plot_data, curr_log, error = None):
        self.curr_iter = curr_iter
        self.conv_val = conv_val
        self.plot_data = plot_data
        self.curr_log = curr_log
        self.error = error

class NNLearner(QObject):
    def __init__(self, nn, fcn, iter_num, range_x, range_y, \
        learn_pause_obj, learn_stop_obj, learn_state_mutex):
        QObject.__init__(self)
        self._nn = nn
        self._fcn = fcn
        self._iter_num = iter_num
        self._range_x = range_x
        self._range_y = range_y
        self._x_min = np.min(range_x)
        self._x_max = np.max(range_x)
        self._y_min = np.min(range_y)
        self._y_max = np.max(range_y)
        self._learning_paused = learn_pause_obj # external
        self._learning_stopped = learn_stop_obj # external
        self._learn_state_mutex = learn_state_mutex # external
        self._pause_wait_mutex = QMutex()

    @pyqtSlot()
    def learn(self):
        rand_gen = rand.default_rng(None)

        plot_ins = np.empty((2, 0))
        for i in range(len(self._range_x)):
            for j in range(len(self._range_y)):
                plot_ins = np.hstack((plot_ins, \
                    np.array([[self._range_x[i]], [self._range_y[j]]])))
        
        x = rand_gen.uniform(self._x_min, self._x_max, \
            size = (1, self._iter_num)) # array of example_ins[0]
        y = rand_gen.uniform(self._y_min, self._y_max, \
            size = (1, self._iter_num)) # array of example_ins[1]
        example_ins = np.vstack((x, y)).copy()

        try:
            example_outs = ne.evaluate(self._fcn)
        except Exception as ex:
            self._return_with_error(str(ex))
            return

        x = plot_ins[0, :]
        y = plot_ins[1, :]
        try:
            ref_plot_outs = ne.evaluate(self._fcn)
        except Exception as ex:
            self._return_with_error(str(ex))
            return
        
        ref_plot_outs = np.reshape(ref_plot_outs, (len(self._range_y), len(self._range_x)), "F")

        for i in range(1, self._iter_num + 1):
            self._learn_state_mutex.lock()
            while (self._learning_paused.value):
                self._learn_state_mutex.unlock()
                QThread.msleep(20)
                self._learn_state_mutex.lock()
            if (self._learning_stopped.value):
                self._learn_state_mutex.unlock()
                self.done.emit()
                return
            self._learn_state_mutex.unlock()

            nn_ins = self._scale_inputs(example_ins[:, i - 1])
            nn_outs = self._nn.calc_outs(nn_ins)
            self._nn.backprop(example_outs[:, i - 1])
            conv_val = self._nn.update_weights()

            plot_outs = self._nn.calc_outs(self._scale_inputs(plot_ins))
            plot_outs = np.reshape(plot_outs, (len(self._range_y), len(self._range_x)), "F")
            plot_data = ContourPlotData(self._range_x, self._range_y, plot_outs, ref_plot_outs)
            
            curr_log = "x = %.3e, y = %.3e, f_ref = %.3e, " \
                "f_calc = %.3e, conv_val = %.3e" % \
                (example_ins[0, i - 1], example_ins[1, i - 1], example_outs[0, i - 1], \
                nn_outs[0], conv_val)
            self.send_curr_state.emit(NNState(i, conv_val, plot_data, curr_log))
        self.done.emit()

    def _return_with_error(self, error):
        self.send_curr_state.emit(NNState(None, None, None, None, error))
        self.done.emit()

    def _scale_inputs(self, inputs):
        res = NeuralNetwork.expand_array(inputs.copy())
        res[0, :] = \
            (res[0, :] - self._x_min) / (self._x_max - self._x_min) * 2.0 - 1.0
        res[1, :] = \
            (res[1, :] - self._y_min) / (self._y_max - self._y_min) * 2.0 - 1.0
        return res

    send_curr_state = pyqtSignal(NNState)
    done = pyqtSignal()

class NNGUIMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self._min_coord_value = -2.0
        self._max_coord_value = 2.0
        self._min_coord_value_str = str(self._min_coord_value)
        self._max_coord_value_str = str(self._max_coord_value)

        # interface building
        # level 0
        self._layout_main = QHBoxLayout()
        #     level 1
        self._layout_setup = QVBoxLayout()
        #         level 2
        self._box_fcn = QGroupBox("Function to fit (numexpr library style) (edit = reset)")
        #             level 3
        self._layout_fcn = QVBoxLayout()
        #                 level 4
        self._layout_fcn_text = QHBoxLayout()
        #                     level 5
        self._label_fcn = QLabel("f(x, y) = ")
        self._edit_fcn = QLineEdit("(1.0 + tanh((x + y) / 2.0)) / 2.0")
        self._layout_fcn_text.addWidget(self._label_fcn)
        self._layout_fcn_text.addWidget(self._edit_fcn)
        #                 level 4
        self._layout_x_range = RangeEdit("x", \
            self._min_coord_value_str, self._max_coord_value_str)
        self._layout_y_range = RangeEdit("y", \
            self._min_coord_value_str, self._max_coord_value_str)
        self._layout_fcn.addLayout(self._layout_fcn_text)
        self._layout_fcn.addLayout(self._layout_x_range)
        self._layout_fcn.addLayout(self._layout_y_range)
        #             level 3
        self._box_fcn.setLayout(self._layout_fcn)
        self._box_fcn.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, \
            QSizePolicy.Policy.Minimum)
        #         level 2
        self._layout_settings = QHBoxLayout()
        #             level3
        self._box_arch = QGroupBox("NN inner layers (edit = reset)")
        #                 level 4
        self._layout_arch = QVBoxLayout()
        #                     level 5
        self._table_arch = NNInnerStructureTable()
        self._layout_arch.addWidget(self._table_arch)
        #             level 3
        self._box_arch.setLayout(self._layout_arch)
        self._box_arch.setSizePolicy(QSizePolicy.Policy.Preferred, \
            QSizePolicy.Policy.MinimumExpanding)
        self._box_learn_str = QGroupBox("Learning strength (edit = reset)")
        #                 level 4
        self._layout_learn_str = NNLearningStrengthControl("0.1 + 0.9 / i ** 0.3")
        #             level 3
        self._box_learn_str.setLayout(self._layout_learn_str)
        self._layout_settings.addWidget(self._box_arch)
        self._layout_settings.addWidget(self._box_learn_str)
        #         level 2
        self._box_learn_ctrl = QGroupBox("Learning controls")
        #             level 3
        self._layout_learn_ctrl = QHBoxLayout()
        #                 level 4
        self._label_iter_n = QLabel("# of iterations\n(1-1,000,000)")
        self._edit_iter_n = AutoLimitIntLineEdit("10000", 1, 1_000_000)
        self._stack_learn_pause = QStackedWidget()
        #                     level 5
        self._btn_learn = QPushButton("Learn")
        self._btn_pause_toggle = QPushButton("Pause")
        #                 level 4
        self._stack_learn_pause.addWidget(self._btn_learn)
        self._stack_learn_pause.addWidget(self._btn_pause_toggle)
        self._stack_reset_stop = QStackedWidget()
        #                     level 5
        self._btn_reset = QPushButton("Reset")
        self._btn_stop = QPushButton("Stop")
        #                 level 4
        self._stack_reset_stop.addWidget(self._btn_reset)
        self._stack_reset_stop.addWidget(self._btn_stop)
        #             level 3
        self._layout_learn_ctrl.addWidget(self._label_iter_n)
        self._layout_learn_ctrl.addWidget(self._edit_iter_n)
        self._layout_learn_ctrl.addStretch()
        self._layout_learn_ctrl.addWidget(self._stack_learn_pause)
        self._layout_learn_ctrl.addWidget(self._stack_reset_stop)
        #         level 2
        self._box_learn_ctrl.setLayout(self._layout_learn_ctrl)
        self._box_log = QGroupBox("Log")
        #             level 3
        self._layout_log = QVBoxLayout()
        #                 level 4
        self._edit_log = QPlainTextEdit()
        self._edit_log.setReadOnly(True)
        self._edit_log.setMaximumBlockCount(100_000)
        self._layout_log.addWidget(self._edit_log)
        #         level 2
        self._box_log.setLayout(self._layout_log)
        self._layout_setup.addWidget(self._box_fcn)
        self._layout_setup.addLayout(self._layout_settings)
        self._layout_setup.addWidget(self._box_learn_ctrl)
        self._layout_setup.addWidget(self._box_log)
        self._layout_setup.setStretchFactor(self._box_fcn, 0)
        self._layout_setup.setStretchFactor(self._layout_settings, 3)
        self._layout_setup.setStretchFactor(self._box_learn_ctrl, 0)
        self._layout_setup.setStretchFactor(self._box_log, 2)
        #     level 1
        self._layout_plots = QVBoxLayout()
        #         level 2
        self._box_conv = QGroupBox("Convergence")
        #             level 3
        self._layout_conv = QVBoxLayout()
        #                 level 4
        self._cbox_conv = QCheckBox("Live update")
        self._cbox_conv.setChecked(True)
        self._conv_plot_handler = PlotHandler(ConvPlotUpdater)
        self._fig_canv_conv = self._conv_plot_handler.get_canvas()
        self._navbar_conv = self._conv_plot_handler.add_navbar_for_window(self)
        self._layout_conv.addWidget(self._cbox_conv)
        self._layout_conv.addWidget(self._navbar_conv)
        self._layout_conv.addWidget(self._fig_canv_conv)
        #         level 2
        self._box_conv.setLayout(self._layout_conv)
        self._box_contour = QGroupBox("Contour plot")
        #             level 3
        self._layout_contour = QVBoxLayout()
        #                 level 4
        self._layout_contour_mode = QHBoxLayout()
        self._bgrp_contour_mode = QButtonGroup()
        #                     level 5
        self._rbtn_plot_fit = QRadioButton("NN fit")
        self._rbtn_plot_fcn = QRadioButton("Function")
        self._rbtn_plot_err = QRadioButton("Rel. error (log(1+ε))")
        self._layout_contour_mode.addWidget(self._rbtn_plot_fit)
        self._layout_contour_mode.addWidget(self._rbtn_plot_fcn)
        self._layout_contour_mode.addWidget(self._rbtn_plot_err)
        self._bgrp_contour_mode.addButton(self._rbtn_plot_fit)
        self._bgrp_contour_mode.addButton(self._rbtn_plot_fcn)
        self._bgrp_contour_mode.addButton(self._rbtn_plot_err)
        self._rbtn_plot_fit.setChecked(True)
        #                 level 4
        self._cbox_contour = QCheckBox("Live update")
        self._cbox_contour.setChecked(True)
        self._contour_plot_handler = PlotHandler(ContourPlotUpdater)
        self._fig_canv_contour = self._contour_plot_handler.get_canvas()
        self._navbar_contour = self._contour_plot_handler.add_navbar_for_window(self)
        self._layout_contour.addLayout(self._layout_contour_mode)
        self._layout_contour.addWidget(self._cbox_contour)
        self._layout_contour.addWidget(self._navbar_contour)
        self._layout_contour.addWidget(self._fig_canv_contour)
        #         level 2
        self._box_contour.setLayout(self._layout_contour)
        self._layout_plots.addWidget(self._box_conv)
        self._layout_plots.addWidget(self._box_contour)
        #     level 1
        self._box_struct_disp = QGroupBox("NN structure visualization")
        #         level 2
        self._layout_struct_disp = QVBoxLayout()
        #             level 3
        self._nnd_struct = NNDisplayer()
        self._gview_struct = self._nnd_struct.get_gview()
        self._layout_struct_xy_controls = QHBoxLayout()
        #                 level 4
        avg_coord_value = (self._min_coord_value + self._max_coord_value) / 2.0
        self._dss_x = DoubleSliderSpinbox("x =", \
            avg_coord_value, self._min_coord_value, self._max_coord_value)
        self._dss_y = DoubleSliderSpinbox("y =", \
            avg_coord_value, self._min_coord_value, self._max_coord_value)
        self._layout_struct_xy_controls.addLayout(self._dss_x)
        self._layout_struct_xy_controls.addLayout(self._dss_y)
        #             level 3
        self._layout_struct_disp.addWidget(self._gview_struct)
        self._layout_struct_disp.addLayout(self._layout_struct_xy_controls)
        #         level 2
        self._box_struct_disp.setLayout(self._layout_struct_disp)
        #     level 1
        self._layout_main.addLayout(self._layout_setup)
        self._layout_main.addLayout(self._layout_plots)
        self._layout_main.addWidget(self._box_struct_disp)
        self._layout_main.setStretchFactor(self._layout_setup, 1)
        self._layout_main.setStretchFactor(self._layout_plots, 1)
        self._layout_main.setStretchFactor(self._box_struct_disp, 1)
        # interface building end

        self._central_widget = QWidget()
        self._central_widget.setLayout(self._layout_main)
        self.setCentralWidget(self._central_widget)

        self.resize(1024, 768)
        self.setWindowTitle("NN learning of 2D functions")

        self._edit_fcn.editingFinished.connect(self._reset)
        self._layout_x_range.range_changed_no_arg.connect(self._reset)
        self._layout_y_range.range_changed_no_arg.connect(self._reset)
        self._table_arch.struct_changed.connect(self._reset)
        self._layout_learn_str.learn_str_fcn_changed.connect(self._reset)

        self._btn_learn.pressed.connect(self._learn)
        self._btn_reset.pressed.connect(self._reset)
        self._btn_pause_toggle.pressed.connect(self._pause_toggle)
        self._btn_stop.pressed.connect(self._stop)

        self._nn_learner = None
        self._nn_learner_thread = None
        self._nn_learner_running = False

        self._learning_paused = MutWrap(False)
        self._learning_stopped = MutWrap(False)
        self._learn_state_mutex = QMutex()

        self._reset_conv_plot.connect(self._conv_plot_handler.reset_plot)
        self._show_conv_plot.connect(self._conv_plot_handler.show_plot)

        self._reset_contour_plot.connect(self._contour_plot_handler.reset_plot)
        self._show_contour_plot.connect(self._contour_plot_handler.show_plot)

        self._bgrp_contour_mode.buttonClicked.connect(self._contour_plot_mode_change)

        self._layout_x_range.range_changed.connect(self._dss_x.set_range)
        self._layout_y_range.range_changed.connect(self._dss_y.set_range)

        self._dss_x.value_changed_no_arg.connect(self._nnd_minor_update)
        self._dss_y.value_changed_no_arg.connect(self._nnd_minor_update)

        self._reset()

    def resizeEvent(self, event):
        self._nnd_struct.resize_view()
        QMainWindow.resizeEvent(self, event)

    def showEvent(self, event):
        self._nnd_struct.resize_view()
        QMainWindow.showEvent(self, event)

    def _init_nn_if_none(self):
        if (self._nn is not None):
            return
        nn_layer_sizes = self._table_arch.get_layer_sizes()
        nn_learn_str_fcn = self._layout_learn_str.get_strength_function()
        self._nn = NeuralNetwork([2, *nn_layer_sizes, 1], nn_learn_str_fcn)

    @pyqtSlot()
    def _learn(self):
        if (not self._layout_learn_str.is_str_fcn_ok()):
            return

        self._part_lock_ui_before_upd()

        self._stack_learn_pause.setCurrentWidget(self._btn_pause_toggle)
        self._stack_reset_stop.setCurrentWidget(self._btn_stop)
        
        coord_divs = 100
        x_min, x_max = self._layout_x_range.get_range()
        y_min, y_max = self._layout_y_range.get_range()
        range_x = np.linspace(x_min, x_max, coord_divs + 1)
        range_y = np.linspace(y_min, y_max, coord_divs + 1)

        self._curr_run_iter_max = int(self._edit_iter_n.text())
        self._curr_run_curr_iter = 0
        
        self._reset_conv_plot.emit()
        self._reset_contour_plot.emit()

        self._nn_learner = NNLearner(self._nn, self._edit_fcn.text(), \
            self._curr_run_iter_max, range_x, range_y, \
            self._learning_paused, self._learning_stopped, self._learn_state_mutex)
        self._nn_learner_thread = QThread()

        self._nn_learner.send_curr_state.connect(self._process_sent_nn_state)
        self._nn_learner.done.connect(self._update_nn_end)
        self._do_learning.connect(self._nn_learner.learn)

        self._nn_learner.moveToThread(self._nn_learner_thread)
        self._nn_learner_thread.start()
        self._do_learning.emit()
        self._nn_learner_running = True

    @pyqtSlot(NNState)
    def _process_sent_nn_state(self, nn_state):
        # nn_state and members are repeatedly created in a loop and never changed 
        # so no need to lock nn_state

        error = nn_state.error
        if (error is not None):
            html_before = "<b><font color = \"#FF0000\">"
            html_after = "</font></b>"
            self._edit_log.appendHtml(\
                "{0:s}Error in function: {1:s}{2:s}".format(html_before, error, html_after))
            return

        self._curr_run_curr_iter = nn_state.curr_iter

        self._conv_val_log.append(nn_state.conv_val)
        if (self._cbox_conv.isChecked()):
            self._prepare_show_conv_plot()

        self._last_plot_data = nn_state.plot_data
        if (self._cbox_contour.isChecked()):
            self._prepare_show_contour_plot()

        prev_iter_num = len(self._conv_val_log) - self._curr_run_curr_iter

        self._edit_log.appendPlainText("Iter {0:d}/{1:d} [{2:d}/{3:d}]: {4:s}".format(\
            self._curr_run_curr_iter, self._curr_run_iter_max, \
            self._curr_run_curr_iter + prev_iter_num, self._curr_run_iter_max + prev_iter_num, \
            nn_state.curr_log))

    @pyqtSlot()
    def _update_nn_end(self):
        self._nn_learner_thread.quit()
        self._nn_learner_thread.wait()
        self._nn_learner_thread = None
        self._nn_learner = None
        self._nn_learner_running = False

        self._curr_run_iter_max = None
        self._curr_run_curr_iter = None

        self._learning_paused.value = False
        if (self._learning_stopped.value):
            html_before = "<font color = \"#0000FF\">"
            html_after = "</font>"
            self._edit_log.appendHtml(\
                "{0:s}Learning stopped by user{1:s}".format(html_before, html_after))
        self._learning_stopped.value = False

        self._prepare_show_conv_plot()
        self._prepare_show_contour_plot()

        self._stack_learn_pause.setCurrentWidget(self._btn_learn)
        self._stack_reset_stop.setCurrentWidget(self._btn_reset)

        self._nnd_major_update()

        self._part_unlock_ui_after_upd()

    @pyqtSlot()
    def _reset(self):
        self._nn = None
        self._curr_run_iter_max = None
        self._curr_run_curr_iter = None
        self._conv_val_log = []
        self._last_plot_data = None

        self._edit_log.clear()
        self._reset_conv_plot.emit()
        self._reset_contour_plot.emit()

        self._init_nn_if_none()
        self._nnd_major_update(layer_size_upd = True)
    
    @pyqtSlot()
    def _stop(self):
        self._learn_state_mutex.lock()
        self._learning_paused.value = False
        self._learning_stopped.value = True
        self._learn_state_mutex.unlock()

    @pyqtSlot()
    def _pause_toggle(self):
        new_pause_state = not(self._learning_paused.value)
        self._learn_state_mutex.lock()
        self._learning_paused.value = new_pause_state
        self._learn_state_mutex.unlock()
        self._navbar_conv.setEnabled(new_pause_state)
        self._navbar_contour.setEnabled(new_pause_state)
        if (new_pause_state):
            self._btn_pause_toggle.setText("Resume")
        else:
            self._btn_pause_toggle.setText("Pause")

    def _part_lock_ui_before_upd(self):
        self._edit_fcn.setEnabled(False)
        self._layout_x_range.set_line_edits_enabled(False)
        self._layout_y_range.set_line_edits_enabled(False)

        self._table_arch.setEnabled(False)
        self._layout_learn_str.set_enable_edit(False)
        self._edit_iter_n.setEnabled(False)

        self._navbar_conv.setEnabled(False)
        self._navbar_contour.setEnabled(False)

        self._dss_x.set_interaction_enabled(False)
        self._dss_y.set_interaction_enabled(False)

    def _part_unlock_ui_after_upd(self):
        self._edit_fcn.setEnabled(True)
        self._layout_x_range.set_line_edits_enabled(True)
        self._layout_y_range.set_line_edits_enabled(True)

        self._table_arch.setEnabled(True)
        self._layout_learn_str.set_enable_edit(True)
        self._edit_iter_n.setEnabled(True)

        self._navbar_conv.setEnabled(True)
        self._navbar_contour.setEnabled(True)

        self._dss_x.set_interaction_enabled(True)
        self._dss_y.set_interaction_enabled(True)
    
    def _prepare_show_conv_plot(self):
        if (len(self._conv_val_log) == 0):
            return
        
        conv_val_log_filt_range = np.arange(1, len(self._conv_val_log) + 1)
        conv_val_log_opt_count = 100
        if (len(self._conv_val_log) < 2 * conv_val_log_opt_count):
            conv_val_log_filt = np.array(self._conv_val_log)
        else:
            wnd_len = len(self._conv_val_log) // conv_val_log_opt_count
            wnd_fcn = 1.0 - np.fabs(2.0 * (np.arange(wnd_len) + 0.5) / wnd_len - 1.0)
            wnd_fcn /= np.sum(wnd_fcn)
            conv_val_log_filt = np.array(self._conv_val_log)
            add_right = (wnd_len - 1) // 2
            add_left = (wnd_len - 1) - add_right
            conv_val_log_filt = np.pad(conv_val_log_filt, (add_left, add_right), mode = "edge")
            conv_val_log_filt = np.convolve(conv_val_log_filt, wnd_fcn, mode = "valid")
            conv_val_log_filt_range = np.flip(conv_val_log_filt_range[-1::-wnd_len])
            conv_val_log_filt = conv_val_log_filt[conv_val_log_filt_range - 1]
        plot_data_obj = ConvPlotData(conv_val_log_filt_range, conv_val_log_filt)
        self._show_conv_plot.emit(plot_data_obj)

    def _prepare_show_contour_plot(self):
        if (self._last_plot_data is None):
            return

        coord_x = self._last_plot_data.coord_x
        coord_y = self._last_plot_data.coord_y
        nn_out = self._last_plot_data.data_2d
        ref_out = self._last_plot_data.ref_data_2d

        match self._bgrp_contour_mode.checkedButton():
            case self._rbtn_plot_fit:
                data_2d = nn_out
            case self._rbtn_plot_fcn:
                data_2d = ref_out
            case self._rbtn_plot_err:
                data_2d = np.log1p(2.0 * np.abs(nn_out - ref_out) / \
                    (np.abs(nn_out) + np.abs(ref_out)))

        plot_data_obj = ContourPlotData(coord_x, coord_y, data_2d)
        self._show_contour_plot.emit(plot_data_obj)

    def _nnd_major_update(self, layer_size_upd = False):
        if (layer_size_upd):
            self._nnd_struct.update_layer_sizes(self._nn.get_layer_sizes())
        self._nnd_struct.update_weights(self._nn.get_weights())
        self._nnd_minor_update()

    @pyqtSlot()
    def _nnd_minor_update(self):
        range_x = self._layout_x_range.get_range()
        range_y = self._layout_y_range.get_range()
        nn_input_x = \
            (self._dss_x.get_value() - range_x[0]) / (range_x[1] - range_x[0]) * 2.0 - 1.0
        nn_input_y = \
            (self._dss_y.get_value() - range_y[0]) / (range_y[1] - range_y[0]) * 2.0 - 1.0
        nn_inputs = np.reshape(np.array([nn_input_x, nn_input_y]), (2, 1))
        full_outs = self._nn.calc_outs(nn_inputs, full_output = True)
        self._nnd_struct.update_outputs(full_outs)

    @pyqtSlot(QAbstractButton)
    def _contour_plot_mode_change(self, mode):
        self._learn_state_mutex.lock()
        if (self._nn_learner_running and not self._learning_paused):
            self._learn_state_mutex.unlock()
            return
        self._learn_state_mutex.unlock()
        self._prepare_show_contour_plot()
    
    _do_learning = pyqtSignal()
    _reset_conv_plot = pyqtSignal()
    _reset_contour_plot = pyqtSignal()
    _show_conv_plot = pyqtSignal(ConvPlotData)
    _show_contour_plot = pyqtSignal(ContourPlotData)

app = QApplication(sys.argv)
wnd = NNGUIMainWindow()
wnd.show()
sys.exit(app.exec())
