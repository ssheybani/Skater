import random
import numpy as np

from skater.util import exceptions

try:
    import plotly.offline as py
    py.init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    from plotly import tools
except ImportError:
    raise exceptions.plotlyUnavailableError("plotly is required but unavailable on your system.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors as mcolors
except ImportError:
    raise exceptions.MatplotlibUnavailableError("Matplotlib is required but unavailable on your system.")

_enable_axis = lambda ax, flag: ax.axis("on") if flag is True else ax.axis("off")


def _create_meshgrid(xx, yy, plot_step=0.02):
    xmin, xmax = xx.min() - 0.5, xx.max() + 0.5
    ymin, ymax = yy.min() - 0.5, yy.max() + 0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                         np.arange(ymin, ymax, plot_step))
    return xx, yy


def _generate_contours(est, xx, yy, color_map, ax, **params):
    Z = est.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cf = ax.contourf(xx, yy, Z, alpha=0.8, cmap=color_map, **params)
    return cf


# Reference: https://plot.ly/scikit-learn/plot-voting-decision-regions/
def interactive_plot(est, X0, X1, Y, x_label="X1", y_label="X2", file_name='decision_iplot'):
    figure = tools.make_subplots(rows=1, cols=1, print_grid=False)

    xx, yy = _create_meshgrid(X0, X1)
    Z = est.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # generate the contour
    trace1 = go.Contour(x=xx[0], y=yy[:, 1], z=Z, colorscale='Viridis', opacity=0.2, showscale=False)

    # plot the points
    trace2 = go.Scatter(x=X0, y=X1, showlegend=False, mode='markers',
                        marker=dict(color=Y, line=dict(color='black', width=1),
                                    colorscale='Viridis', showscale=True))

    figure.append_trace(trace1, 1, 1)
    figure.append_trace(trace2, 1, 1)

    layout = go.Layout(
        xaxis=dict(autorange=True,
                   showgrid=False,
                   zeroline=False,
                   showline=True,
                   ticks='',
                   showticklabels=True,
                   title=x_label),
        yaxis=dict(autorange=True,
                   showgrid=False,
                   zeroline=False,
                   showline=True,
                   ticks='',
                   showticklabels=True,
                   title=y_label),
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )

    figure.update(layout=layout)
    py.iplot(figure, filename=file_name)
    return py, figure


def plot_decision_boundary(est, X0, X1, Y, color_map=None, mode='static', random_state=0,
                           width=12, height=10, title='decision_boundary', x0_label='X1', x1_label='X2',
                           enable_axis=False, file_name='decision_plot', **params):
    if mode == 'static':
        colors = list(mcolors.CSS4_COLORS.keys())
        n_classes = len(np.unique(Y))
        random.seed(random_state)
        random.shuffle(colors)
        color_list = colors[:n_classes]
        cm = ListedColormap(color_list) if color_map is None else ListedColormap(color_map)

        X_grid, y_grid = _create_meshgrid(X0, X1)
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        cs = _generate_contours(est, X_grid, y_grid, cm, ax, **params)
        # set other properties of the plot
        ax.scatter(X0, X1, c=Y, cmap=cm, alpha=0.6, linewidths=0.9, edgecolors='white')
        ax.set_xlim(X_grid.min(), X_grid.max())
        ax.set_ylim(y_grid.min(), y_grid.max())
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)
        ax.set_title(title)
        _enable_axis(ax, enable_axis)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        fig.savefig('{}.png'.format(file_name))
        return fig, ax
    else:  # interactive mode
        return interactive_plot(est, X0, X1, Y, x0_label, x1_label, file_name)
