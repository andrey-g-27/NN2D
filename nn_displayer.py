# Classes for visualizing structure of a neural network.

import math as m

from PyQt6.QtWidgets import \
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsTextItem, \
    QGraphicsRectItem
from PyQt6.QtCore import Qt, QObject, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QColorConstants, QPen, QBrush, QFont, QFontDatabase

from neural_network import NeuralNetwork

class NNDispWorker(QObject):
    @staticmethod
    def _color_from_value(value, val_range = None):
        color = QColor()

        if (val_range is not None):
            range_mid = (val_range[0] + val_range[1]) / 2.0
            value_norm = 2.0 * (value - range_mid) / (val_range[1] - val_range[0])
        else:
            value_norm = m.log(1.0 + m.fabs(value)) / 5.0
            if (value < 0.0):
                value_norm = -value_norm
            
        value_norm = max(min(value_norm, 1.0), -1.0)
        hue = 0.0 if (value_norm < 0.0) else (1.0 / 3.0)
        sat = m.fabs(value_norm)
        val = 0.75 + 0.25 * sat
        color.setHsvF(hue, sat, val, 1.0)
        return color
    
    def __init__(self, scene, base_width, base_height):
        QObject.__init__(self)

        self._scene = scene
        self._scene.setBackgroundBrush(QColorConstants.White)
        self._view_width = base_width
        self._view_height = base_height

        self._layer_sizes = []
        self._weights = []
        self._outputs = []

        self._nodes = []
        self._node_labels = []
        self._connections = []
        self._connection_labels = []

        self._base_pen = QPen(QColorConstants.Black)
        self._base_brush = QBrush(QColorConstants.White)
        self._base_font = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))

    @pyqtSlot(list, list, list)
    def update_data(self, layer_sizes, weights, outputs):
        if (len(layer_sizes) > 0):
            self._update_structure(layer_sizes)
        if (len(weights) > 0):
            self._update_weights(weights)
        if (len(outputs) > 0):
            self._update_outputs(outputs)
        self.update_done.emit()

    @pyqtSlot(QRectF)
    def resize_scene(self, new_geom):
        self._view_width = new_geom.width()
        self._view_height = new_geom.height()
        self._scene.setSceneRect(new_geom)

        self._position_nodes()
        self._position_connections()
        self._update_weights()
        self._update_outputs()

    def _update_structure(self, layer_sizes):
        self._layer_sizes = layer_sizes

        self._scene.clear()
        self._create_nodes()
        self._create_connections()

        self._position_nodes()
        self._position_connections()

    def _update_weights(self, weights = None):
        if (weights is None):
            weights = self._weights
        else:
            self._weights = weights
        if (len(weights) == 0):
            return
        
        for prev_layer_idx in range(len(self._nodes) - 1):
            next_layer_idx = prev_layer_idx + 1
            prev_layer_sz = len(self._nodes[prev_layer_idx])
            next_layer_sz = len(self._nodes[next_layer_idx])
            if (next_layer_idx != len(self._nodes) - 1):
                next_layer_sz -= 1
            cur_conn_layer = self._connections[prev_layer_idx]
            cur_conn_layer_labels = self._connection_labels[prev_layer_idx]
            cur_weights_layer = weights[prev_layer_idx].T
            for prev_layer_item_idx in range(prev_layer_sz):
                curr_conn_from = cur_conn_layer[prev_layer_item_idx]
                curr_conn_from_labels = cur_conn_layer_labels[prev_layer_item_idx]
                cur_weights_from = cur_weights_layer[prev_layer_item_idx]
                for next_layer_item_idx in range(next_layer_sz):
                    cur_conn = curr_conn_from[next_layer_item_idx]
                    cur_conn_label = curr_conn_from_labels[next_layer_item_idx]
                    cur_weight = cur_weights_from[next_layer_item_idx]

                    cur_brush = cur_conn.brush()
                    cur_brush.setColor(NNDispWorker._color_from_value(cur_weight))
                    cur_conn.setBrush(cur_brush)

                    cur_conn_label.setPlainText("{0:.3e}".format(cur_weight))

    def _update_outputs(self, outputs = None):
        if (outputs is None):
            outputs = self._outputs
        else:
            self._outputs = outputs
        if (len(outputs) == 0):
            return

        circ_sz = self._calc_circ_sz()

        for layer_idx in range(len(self._nodes)):
            layer_size = len(self._nodes[layer_idx])

            cur_nodes = self._nodes[layer_idx]
            cur_labels = self._node_labels[layer_idx]
            cur_outputs = outputs[layer_idx]

            for item_idx in range(layer_size):
                cur_node = cur_nodes[item_idx]
                cur_node_label = cur_labels[item_idx]
                cur_output = cur_outputs[item_idx]
                
                cur_brush = cur_node.brush()
                cur_brush.setColor(\
                    NNDispWorker._color_from_value(cur_output, [-1.0, 1.0]))
                cur_node.setBrush(cur_brush)

                if ((layer_idx != len(self._nodes) - 1) and (item_idx == layer_size - 1)):
                    num_str = "{0:10.1f}".format(cur_output) # 10 spaces - "fix" for misalignment
                else:
                    num_str = "{0:.3e}".format(cur_output)
                num_str_len = len(num_str)
                cur_node_label.setPlainText(num_str)

    def _create_nodes(self):
        self._nodes = []
        self._node_labels = []
        
        for layer_idx in range(len(self._layer_sizes)):
            layer_size = self._layer_sizes[layer_idx]

            cur_nodes = []
            cur_labels = []

            for item_idx in range(layer_size):
                new_node = QGraphicsEllipseItem()
                new_node_label = QGraphicsTextItem(\
                    "{0:d}, {1:d}".format(layer_idx, item_idx))
                new_node.setZValue(1.0)
                new_node_label.setZValue(3.0)
                new_node.setPen(QPen(self._base_pen))
                new_node.setBrush(QBrush(self._base_brush))
                new_node_label.setFont(self._base_font)
                new_node_label.setDefaultTextColor(QColorConstants.Black)

                self._scene.addItem(new_node)
                self._scene.addItem(new_node_label)
                cur_nodes.append(new_node)
                cur_labels.append(new_node_label)
            
            if (layer_idx != len(self._layer_sizes) - 1):
                ex_node = QGraphicsEllipseItem()
                ex_node_label = QGraphicsTextItem(\
                    "{0:d}, {1:d}".format(layer_idx, layer_size))
                ex_node.setZValue(1.0)
                ex_node_label.setZValue(3.0)
                ex_node.setPen(QPen(self._base_pen))
                ex_node.setBrush(QBrush(self._base_brush))
                ex_node_label.setFont(self._base_font)
                ex_node_label.setDefaultTextColor(QColorConstants.Black)

                self._scene.addItem(ex_node)
                self._scene.addItem(ex_node_label)
                cur_nodes.append(ex_node)
                cur_labels.append(ex_node_label)
            
            self._nodes.append(cur_nodes)
            self._node_labels.append(cur_labels)

    def _position_nodes(self):
        max_layer_size = max(self._layer_sizes) + 1
        layer_count = len(self._layer_sizes)
        if (layer_count == 0):
            layer_count = 1
        circ_sz = 0.5 * min(self._view_width / max_layer_size, \
            self._view_height / layer_count)
        font_sz = int(m.ceil(circ_sz / 6.0))

        y_step = 1.0 / layer_count * self._view_height
        y_base = 1.0 * self._view_height - y_step / 2.0 - circ_sz / 2.0

        item_y = y_base

        for layer_idx in range(len(self._layer_sizes)):
            layer_size = self._layer_sizes[layer_idx]
            x_step = 1.0 / (layer_size + 1) * self._view_width
            if (layer_idx == len(self._layer_sizes) - 1):
                x_step = 1.0 / layer_size * self._view_width
            x_base = x_step / 2.0 - circ_sz / 2.0

            item_x  = x_base

            cur_nodes = self._nodes[layer_idx]
            cur_labels = self._node_labels[layer_idx]

            for item_idx in range(layer_size):
                cur_node = cur_nodes[item_idx]
                cur_node_label = cur_labels[item_idx]

                cur_node.setRect(item_x, item_y, circ_sz, circ_sz)

                cur_font = cur_node_label.font()
                cur_font.setPixelSize(font_sz)
                cur_node_label.setFont(cur_font)
                cur_node_label.setPos(item_x, item_y + circ_sz / 2.0 - font_sz)

                item_x += x_step
            
            if (layer_idx != len(self._layer_sizes) - 1):
                cur_node = cur_nodes[layer_size]
                cur_node_label = cur_labels[layer_size]

                cur_node.setRect(item_x + circ_sz / 4.0, item_y + circ_sz / 4.0, \
                    circ_sz / 2.0, circ_sz / 2.0)

                cur_font = cur_node_label.font()
                cur_font.setPixelSize(font_sz)
                cur_node_label.setFont(cur_font)

                cur_node_label.setPos(item_x, item_y + circ_sz / 2.0 - font_sz)

            item_y -= y_step

    def _create_connections(self):
        self._connections = []
        self._connection_labels = []

        for prev_layer_idx in range(len(self._nodes) - 1):
            next_layer_idx = prev_layer_idx + 1
            prev_layer_sz = len(self._nodes[prev_layer_idx])
            next_layer_sz = len(self._nodes[next_layer_idx])
            if (next_layer_idx != len(self._nodes) - 1):
                next_layer_sz -= 1
            cur_conn_layer = []
            cur_conn_layer_labels = []
            for prev_layer_item_idx in range(prev_layer_sz):
                curr_conn_from = []
                curr_conn_from_labels = []
                for next_layer_item_idx in range(next_layer_sz):
                    cur_conn = QGraphicsRectItem()
                    cur_conn_label = QGraphicsTextItem(\
                        "{0:s} -> {1:s}".format(\
                            self._node_labels[prev_layer_idx][prev_layer_item_idx].toPlainText(), \
                            self._node_labels[next_layer_idx][next_layer_item_idx].toPlainText()))
                    cur_conn.setZValue(0.0)
                    cur_conn_label.setZValue(2.0)
                    cur_conn.setPen(QPen(self._base_pen))
                    cur_conn.setBrush(QBrush(self._base_brush))
                    cur_conn_label.setFont(self._base_font)
                    cur_conn_label.setDefaultTextColor(QColorConstants.Black)

                    self._scene.addItem(cur_conn)
                    self._scene.addItem(cur_conn_label)
                    curr_conn_from.append(cur_conn)
                    curr_conn_from_labels.append(cur_conn_label)
                cur_conn_layer.append(curr_conn_from)
                cur_conn_layer_labels.append(curr_conn_from_labels)
            self._connections.append(cur_conn_layer)
            self._connection_labels.append(cur_conn_layer_labels)

    def _position_connections(self):
        circ_sz = self._calc_circ_sz()
        font_sz = int(m.ceil(circ_sz / 8.0))

        for prev_layer_idx in range(len(self._nodes) - 1):
            next_layer_idx = prev_layer_idx + 1
            prev_layer_sz = len(self._nodes[prev_layer_idx])
            next_layer_sz = len(self._nodes[next_layer_idx])
            if (next_layer_idx != len(self._nodes) - 1):
                next_layer_sz -= 1
            cur_conn_layer = self._connections[prev_layer_idx]
            cur_conn_layer_labels = self._connection_labels[prev_layer_idx]
            for prev_layer_item_idx in range(prev_layer_sz):
                curr_conn_from = cur_conn_layer[prev_layer_item_idx]
                curr_conn_from_labels = cur_conn_layer_labels[prev_layer_item_idx]
                for next_layer_item_idx in range(next_layer_sz):
                    cur_conn = curr_conn_from[next_layer_item_idx]
                    cur_conn_label = curr_conn_from_labels[next_layer_item_idx]

                    node_from = self._nodes[prev_layer_idx][prev_layer_item_idx]
                    node_to = self._nodes[next_layer_idx][next_layer_item_idx]
                    node_from_rect = node_from.rect()
                    node_to_rect = node_to.rect()
                    line_x1 = node_from_rect.x() + node_from_rect.width() / 2.0
                    line_y1 = node_from_rect.y() + node_from_rect.height() / 2.0
                    line_x2 = node_to_rect.x() + node_to_rect.width() / 2.0
                    line_y2 = node_to_rect.y() + node_to_rect.height() / 2.0
                    line_dx = line_x2 - line_x1
                    line_dy = line_y2 - line_y1
                    line_len = m.dist((line_x1, line_y1), (line_x2, line_y2))
                    shift_x = line_dy / line_len * circ_sz / 12.0
                    shift_y = -line_dx / line_len * circ_sz / 12.0
                    rot_ang = m.degrees(m.atan2(line_dy, line_dx))

                    cur_conn.setRect(0.0, 0.0, line_len, circ_sz / 6.0)
                    cur_conn.setPos(line_x1 + shift_x, line_y1 + shift_y)
                    cur_conn.setRotation(rot_ang)

                    text_rot_ang = rot_ang
                    text_pos_x = line_x2 - (circ_sz / 2.0 + circ_sz * 0.75) * m.cos(m.radians(text_rot_ang)) + shift_x * 1.67
                    text_pos_y = line_y2 - (circ_sz / 2.0 + circ_sz * 0.75) * m.sin(m.radians(text_rot_ang)) + shift_y * 1.67
                    if (m.fabs(text_rot_ang) > 90.0):
                        if (text_rot_ang > 0.0):
                            text_rot_ang -= 180.0
                        else:
                            text_rot_ang += 180.0
                        text_pos_x = line_x2 + circ_sz / 2.0 * m.cos(m.radians(text_rot_ang)) - shift_x * 1.67
                        text_pos_y = line_y2 + circ_sz / 2.0 * m.sin(m.radians(text_rot_ang)) - shift_y * 1.67
                    cur_font = cur_conn_label.font()
                    cur_font.setPixelSize(font_sz)
                    cur_conn_label.setFont(cur_font)
                    cur_conn_label.setPos(text_pos_x, text_pos_y)
                    cur_conn_label.setRotation(text_rot_ang)

    def _calc_circ_sz(self):
        max_layer_size = max(self._layer_sizes) + 1
        layer_count = len(self._layer_sizes)
        if (layer_count == 0):
            layer_count = 1
        circ_sz = 0.5 * min(self._view_width / max_layer_size, \
            self._view_height / layer_count)
        return circ_sz

    update_done = pyqtSignal()

class NNDisplayer(QObject):
    def __init__(self):
        QObject.__init__(self)

        self._layer_sizes = []
        self._weights = []
        self._outputs = []

        self._is_busy = False
        self._next_layer_sizes = []
        self._next_weights = []
        self._next_outputs = []

        self._gscene = QGraphicsScene()
        self._gview = QGraphicsView(self._gscene)

        self._worker = NNDispWorker(self._gscene, \
            self._gview.geometry().width(), self._gview.geometry().height())

        self._update_data.connect(self._worker.update_data, \
            type = Qt.ConnectionType.QueuedConnection)
        self._worker.update_done.connect(self._update_done, \
            type = Qt.ConnectionType.QueuedConnection)
        
        self._resize_scene.connect(self._worker.resize_scene, \
            type = Qt.ConnectionType.QueuedConnection)

    def get_gview(self):
        return self._gview

    def resize_view(self):
        new_geom = self._gview.geometry()
        self._resize_scene.emit(QRectF(0.0, 0.0, \
            new_geom.width() - new_geom.x(), new_geom.height() - new_geom.y()))

    @pyqtSlot(NeuralNetwork)
    def update_layer_sizes(self, new_layer_sizes):
        self._weights = []
        self._outputs = []
        self._next_weights = []
        self._next_outputs = []
        if (self._is_busy):
            self._next_layer_sizes = new_layer_sizes
            return
        self._is_busy = True
        self._update_layer_sizes(new_layer_sizes)
    
    def _update_layer_sizes(self, new_layer_sizes):
        self._layer_sizes = new_layer_sizes
        self._weights = self._next_weights
        self._outputs = self._next_outputs
        self._next_weights = []
        self._next_outputs = []
        self._update_data.emit(new_layer_sizes, self._weights, self._outputs)

    @pyqtSlot()
    def update_weights(self, weights):
        self._outputs = []
        self._next_outputs = []
        if (self._is_busy):
            self._next_weights = weights
            return
        self._is_busy = True
        self._update_weights(weights)

    def _update_weights(self, weights):
        self._weights = weights
        self._outputs = self._next_outputs
        self._next_outputs = []
        self._update_data.emit([], weights, self._outputs)

    @pyqtSlot(list)
    def update_outputs(self, outputs):
        if (self._is_busy):
            self._next_outputs = outputs
            return
        self._is_busy = True
        self._update_outputs(outputs)
    
    def _update_outputs(self, outputs):
        self._outputs = outputs
        self._update_data.emit([], [], outputs)

    @pyqtSlot()
    def _update_done(self):
        if (len(self._next_layer_sizes) > 0):
            self._update_layer_sizes(self._next_layer_sizes)
            self._next_layer_sizes = []
            return
        if (len(self._next_weights) > 0):
            self._update_weights(self._next_weights)
            self._next_weights = []
            return
        if (len(self._next_outputs) > 0):
            self._update_outputs(self._next_outputs)
            self._next_outputs = []
            return
        self._is_busy = False

    _update_data = pyqtSignal(list, list, list)
    _resize_scene = pyqtSignal(QRectF)
