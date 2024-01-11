# NN2D
Learning 2D mathematical functions with a neural network (NN). It was created out of desire to understand how neural networks operate but also to further the understanding of Qt.

Features include changing function and coordinate range to learn, ability to set NN structure and learning rate, visualizations of function and NN output during learning, visualization of NN structure with color-coded weights and output values.

## Project files
* `nn2d_gui.py` - main program with GUI.
* `neural_network.py` - module with NN code (including backpropagation).
* `custom_controls.py` - several QT-based controls for interacting with NN.
* `threaded_figure.py` - module for threaded visualizations of variety of data.
* `nn_displayer.py` - module for visualizing internal structure of NN.

## Required libraries
PyQt6, NumPy, Numexpr, Matplotlib.
