import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animation_test(i, data_plot_in, ax_in, lines):
    lines.set_data(data_plot_in[0,0:i],data_plot_in[1,0:i])
    return lines

