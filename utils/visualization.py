from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def create_figure(width=500, height=500, onScreen=False):
    """
    Create a figure in memory or on-screen, and resize the figure to a specific resolution

    """

    if onScreen:
        fig = plt.figure()
    else:
        fig = Figure()

    # resize the figure
    DPI = fig.get_dpi()
    fig.set_size_inches(float(width) / float(DPI), float(height) / float(DPI))

    return fig