# Classes for visualizing data via matplotlib.
# Use threads (Qt) to avoid slowing down main thread.

from enum import Enum
import math as m

import numpy as np

import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.pyplot as pp

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from matplotlib.backends.backend_qtagg import \
(FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavBar)

class ConvPlotData:
    def __init__(self, labels, values):
        self.labels = labels
        self.values = values

class ContourPlotData:
    def __init__(self, coord_x, coord_y, data_2d, ref_data_2d = None):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.data_2d = data_2d
        self.ref_data_2d = ref_data_2d

class LearningStrengthPlotData:
    def __init__(self, i_values, s_values):
        self.i_values = i_values
        self.s_values = s_values

class ConvPlotUpdater(QObject):
    def __init__(self, plot_fig, plot_canv):
        QObject.__init__(self)
        self._plot_fig = plot_fig
        self._plot_canv = plot_canv
        self._plot_axes = None
        self._min_y = None
        self._max_y = None

    @pyqtSlot()
    def reset_plot(self):
        self._min_y = None
        self._max_y = None
        self._basic_reset()
        self._plot_axes.set_ylim(1.0e-2, 1.0e+0)
        self._plot_canv.draw()
        self.reset_done.emit()
    
    @pyqtSlot(ConvPlotData)
    def show_plot(self, plot_data):
        self._basic_reset()

        labels = plot_data.labels
        values = plot_data.values

        if (self._min_y is None):
            self._min_y = np.amin(values)
        else:
            self._min_y = min(self._min_y, np.amin(values))
        
        if (self._max_y is None):
            self._max_y = np.amax(values)
        else:
            self._max_y = max(self._max_y, np.amax(values))
        
        self._plot_axes.semilogy(labels, values, "b-")

        lim_rat = self._max_y / self._min_y
        if (lim_rat < 100.0):
            geom_mean = m.sqrt(self._min_y * self._max_y)
            min_y_plot = 0.1 * geom_mean
            max_y_plot = 10.0 * geom_mean
        else:
            lim_ex_mul = pow(lim_rat, 0.167)
            min_y_plot = self._min_y / lim_ex_mul
            max_y_plot = self._max_y * lim_ex_mul
        
        self._plot_axes.set_ylim(min_y_plot, max_y_plot)
        self._plot_canv.draw()
        self.show_done.emit()

    def _basic_reset(self):
        self._plot_fig.clear()
        self._plot_fig.set_tight_layout(True)
        self._plot_axes = self._plot_fig.add_subplot(1, 1, 1)
        self._plot_axes.set_yscale("log")
        self._plot_axes.grid(True)
        self._plot_axes.set_xlabel("Iteration")
        self._plot_axes.set_ylabel("Convergence value")
    
    reset_done = pyqtSignal()
    show_done = pyqtSignal()

class ContourPlotUpdater(QObject):
    def __init__(self, plot_fig, plot_canv):
        QObject.__init__(self)
        self._plot_fig = plot_fig
        self._plot_canv = plot_canv
        self._plot_axes = None

    @pyqtSlot()
    def reset_plot(self):
        self._basic_reset()
        self._plot_canv.draw()
        self.reset_done.emit()
    
    @pyqtSlot(ContourPlotData)
    def show_plot(self, plot_data):
        coord_x = plot_data.coord_x
        coord_y = plot_data.coord_y
        data_2d = plot_data.data_2d

        self._basic_reset()

        data_min = np.amin(data_2d)
        data_max = np.amax(data_2d)
        pl = self._plot_axes.contourf(coord_x, coord_y, data_2d, \
            np.linspace(data_min, data_max, 51), cmap = mpl.cm.jet)
        pp.colorbar(pl, ax = self._plot_axes, \
            ticks = np.linspace(data_min, data_max, 9))
        self._plot_canv.draw()
        self.show_done.emit()

    def _basic_reset(self):
        self._plot_fig.clear()
        self._plot_fig.set_tight_layout(True)
        self._plot_axes = self._plot_fig.add_subplot(1, 1, 1)
        self._plot_axes.set_xlabel("x")
        self._plot_axes.set_ylabel("y")
    
    reset_done = pyqtSignal()
    show_done = pyqtSignal()

class LearningStrengthPlotUpdater(QObject):
    def __init__(self, plot_fig, plot_canv):
        QObject.__init__(self)
        self._plot_fig = plot_fig
        self._plot_canv = plot_canv
        self._plot_axes = None
        self._min_x_default = 1.0e+0
        self._max_x_default = 1.0e+6
        self._min_y_default = 1.0e-4
        self._max_y_default = 1.0e+0

    @pyqtSlot()
    def reset_plot(self):
        self._basic_reset()
        self._plot_axes.set_xlim(self._min_x_default, self._max_x_default)
        self._plot_axes.set_ylim(self._min_y_default, self._max_y_default)
        self._plot_canv.draw()
        self.reset_done.emit()
    
    @pyqtSlot(ConvPlotData)
    def show_plot(self, plot_data):
        self._basic_reset()

        i_values = plot_data.i_values
        s_values = plot_data.s_values

        min_x = np.amin(i_values)
        max_x = np.amax(i_values)
        min_y = np.amin(s_values)
        max_y = np.amax(s_values)

        # placeholders to avoid errors while editing functions
        if (m.isnan(min_x) or (min_x <= 0.0)):
            min_x = self._min_x_default
        if (m.isnan(max_x) or (max_x <= 0.0)):
            max_x = self._max_x_default
        if (m.isnan(min_y) or (min_y <= 0.0)):
            min_y = self._min_y_default
        if (m.isnan(max_y) or (max_y <= 0.0)):
            max_y = self._max_y_default
        
        lim_rat_x = max_x / min_x
        lim_ex_mul_x = pow(lim_rat_x, 0.167)
        min_x_plot = min_x / lim_ex_mul_x
        max_x_plot = max_x * lim_ex_mul_x

        lim_rat_y = max_y / min_y
        if (lim_rat_y < 100.0):
            geom_mean_y = m.sqrt(min_y * max_y)
            min_y_plot = 0.1 * geom_mean_y
            max_y_plot = 10.0 * geom_mean_y
        else:
            lim_ex_mul_y = pow(lim_rat_y, 0.167)
            min_y_plot = min_y / lim_ex_mul_y
            max_y_plot = max_y * lim_ex_mul_y

        if (not m.isfinite(min_x_plot)):
            min_x_plot = self._min_x_default
        if (not m.isfinite(max_x_plot)):
            max_x_plot = self._max_x_default
        if (not m.isfinite(min_y_plot)):
            min_y_plot = self._min_y_default
        if (not m.isfinite(max_y_plot)):
            max_y_plot = self._max_y_default
        
        if (i_values.shape == s_values.shape):
            self._plot_axes.loglog(i_values, s_values, "b-")
        else:
            # placeholder to avoid errors while editing functions
            self._plot_axes.loglog([1.0], [1.0], "b-")
        self._plot_axes.set_xlim(min_x_plot, max_x_plot)
        self._plot_axes.set_ylim(min_y_plot, max_y_plot)
        self._plot_canv.draw()

        self.show_done.emit()

    def _basic_reset(self):
        self._plot_fig.clear()
        self._plot_fig.set_tight_layout(True)
        self._plot_axes = self._plot_fig.add_subplot(1, 1, 1)

        self._plot_axes.set_xscale("log")
        self._plot_axes.set_yscale("log")
        self._plot_axes.grid(True)
        self._plot_axes.set_xlabel("Iteration")
        self._plot_axes.set_ylabel("Learning strength")
    
    reset_done = pyqtSignal()
    show_done = pyqtSignal()

class PHCommandType(Enum):
    PHC_RESET = 0
    PHC_SHOW = 1

class PlotHandlerCommand:
    def __init__(self, command_type, command_data):
        self.type = command_type
        self.data = command_data

class PlotHandler(QObject):
    def __init__(self, updater_class):
        QObject.__init__(self)
        self._plot_fig = pp.figure()
        self._plot_canv = FigureCanvas(self._plot_fig)
        self._navbar = None

        self._is_busy = False
        self._next_command = None

        # variables are used only there so no need to protect them
        self._plot_updater = updater_class(self._plot_fig, self._plot_canv)
        self._plot_updater_thread = QThread()

        self._reset_plot.connect(self._plot_updater.reset_plot)
        self._show_plot.connect(self._plot_updater.show_plot)
        self._plot_updater.reset_done.connect(self._action_finished)
        self._plot_updater.show_done.connect(self._action_finished)
        self._plot_updater.moveToThread(self._plot_updater_thread)
        self._plot_updater_thread.start()

    def get_canvas(self):
        return self._plot_canv

    def add_navbar_for_window(self, window):
        self._navbar = NavBar(self._plot_canv, window)
        return self._navbar
    
    @pyqtSlot()
    def reset_plot(self):
        if (self._is_busy):
            self._next_command = PlotHandlerCommand(PHCommandType.PHC_RESET, None)
            return
        self._is_busy = True
        self._reset_plot.emit()

    @pyqtSlot(object)
    def show_plot(self, data):
        if (self._is_busy):
            self._next_command = PlotHandlerCommand(PHCommandType.PHC_SHOW, data)
            return
        self._is_busy = True
        self._show_plot.emit(data)

    @pyqtSlot()
    def _action_finished(self):
        if (self._next_command is None):
            self._is_busy = False
            return
        match self._next_command.type:
            case PHCommandType.PHC_RESET:
                self._reset_plot.emit()
            case PHCommandType.PHC_SHOW:
                self._show_plot.emit(self._next_command.data)
        self._next_command = None

    _reset_plot = pyqtSignal()
    _show_plot = pyqtSignal(object)
