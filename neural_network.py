# Class for neural network.

import math as m

import numpy as np
import numpy.linalg as la
import numpy.random as rand
import numexpr as ne

class NeuralNetwork:
    # No input checks performed!

    @staticmethod
    def _sigma(x):
        return (1.0 / (1.0 + np.exp(-x)) * 2.0 - 1.0)
    
    @staticmethod
    def _sigma_der(x):
        return (0.5 * (1.0 + NeuralNetwork._sigma(x)) * (1.0 - NeuralNetwork._sigma(x)))

    @staticmethod
    def _deepcopy_list(arg):
        res = []
        for item in arg:
            res.append(item.copy())
        return res

    @staticmethod
    def expand_array(arg):
        arg_shape = arg.shape
        if (len(arg_shape) < 2):
            res = np.expand_dims(arg, axis = 1)
        else:
            res = arg
        return res.copy()

    def __init__(self, layer_sizes, learn_str_fcn):
        self._layer_sizes = layer_sizes
        self.randomize_weights(1.0)
        self._learn_str_fcn = learn_str_fcn
        self._learn_iters = 0
        self._k_norm = 0.0
    
    def randomize_weights(self, magn):
        rand_gen = rand.default_rng(None)
        self._weights = []
        ls = self._layer_sizes
        for i in range(len(ls) - 1):
            self._weights.append(magn * rand_gen.normal(size = (ls[i + 1], ls[i] + 1)))
    
    def get_layer_sizes(self):
        return self._layer_sizes[:]

    def set_weights(self, weights):
        self._weights = NeuralNetwork._deepcopy_list(weights)

    def get_weights(self):
        return NeuralNetwork._deepcopy_list(self._weights)

    def calc_outs(self, ins, full_output = False):
        # batch inputs allowed, but only without learning
        self._ins = ins.copy()
        outs = self.expand_array(ins)
        outs = np.vstack((outs, np.ones((1, outs.shape[1]))))
        outs_cumul = [outs.copy().flatten()]
        acts = []
        f_ders = []
        for i in range(len(self._weights)):
            acts.append(outs.copy())
            cur_sums = self._weights[i] @ outs
            outs = self._sigma(cur_sums)
            f_ders.append(self._sigma_der(cur_sums))
            outs = self.expand_array(outs)
            outs = np.vstack((outs, np.ones((1, outs.shape[1]))))
            outs_cumul_next = outs.copy().flatten()
            outs_cumul.append(outs_cumul_next)
        outs = outs[0 : -1, :]
        self._outs = outs
        self._acts = acts
        self._f_ders = f_ders
        if (full_output):
            return outs_cumul
        return outs.copy()

    def backprop(self, outs_ref):
        self._outs_ref = outs_ref.copy()
        weights_corr = []
        k_e = 1.0
        cur_err = 2.0 * k_e * (self._outs - outs_ref)
        for i in range(len(self._weights) - 1, -1, -1):
            cur_err = self._f_ders[i] * cur_err
            weights_grad = cur_err @ self._acts[i].T
            weights_corr.insert(0, weights_grad)
            cur_err = self._weights[i].T @ cur_err
            cur_err = cur_err[0 : -1, :]
        self._weights_corr = weights_corr

    def update_weights(self):
        self._learn_iters += 1
        i = self._learn_iters
        alpha = ne.evaluate(self._learn_str_fcn)
        if (alpha < 0.0):
            alpha = 0.0

        conv_val = la.norm(2.0 * np.abs(self._outs - self._outs_ref) / \
            (np.abs(self._outs) + np.abs(self._outs_ref)))
        weights_corr_norm = 0.0
        weights_num = 0
        for l in range(len(self._weights)):
            weights_corr_norm = m.sqrt(weights_corr_norm ** 2 + \
                                    la.norm(self._weights_corr[l]) ** 2)
            weights_num = weights_num + np.size(self._weights_corr[l])
        weights_corr_mean_sqr = weights_corr_norm / m.sqrt(weights_num)
        for l in range(len(self._weights)):
            self._weights_corr[l] = \
                self._k_norm * self._weights_corr[l] / weights_corr_mean_sqr + \
                    (1.0 - self._k_norm) * self._weights_corr[l]
        
        for l in range(len(self._weights)):
            self._weights[l] = self._weights[l] - alpha * self._weights_corr[l]
        return conv_val
