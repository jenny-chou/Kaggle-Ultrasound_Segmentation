# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.legend(['val_loss', 'loss'])
    