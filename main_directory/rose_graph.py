import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pprint

def main():
    azi = np.random.uniform(0, 360, 100000)
    print azi.shape
    z = np.cos(np.radians(azi/2.))

    plt.figure(figsize=(5, 6))
    plt.subplot(111, projection='polar')
    coll = rose(azi, z=z, bidirectional=False)
    plt.xticks(np.radians(range(0, 360, 45)),
               ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    plt.colorbar(coll, orientation='horizontal')
    plt.xlabel('A rose diagram colored by a second variable')
    plt.rgrids(range(5, 20, 5), angle=360)

    plt.show()


def rose(azimuths, z=None, ax=None, bins=30, bidirectional=False,
         color_by=np.mean, **kwargs):
    """Create a "rose" diagram (a.k.a. circular histogram).

    Parameters:
    -----------
        azimuths: sequence of numbers
            The observed azimuths in degrees.
        z: sequence of numbers (optional)
            A second, co-located variable to color the plotted rectangles by.
        ax: a matplotlib Axes (optional)
            The axes to plot on. Defaults to the current axes.
        bins: int or sequence of numbers (optional)
            The number of bins or a sequence of bin edges to use.
        bidirectional: boolean (optional)
            Whether or not to treat the observed azimuths as bi-directional
            measurements (i.e. if True, 0 and 180 are identical).
        color_by: function or string (optional)
            A function to reduce the binned z values with. Alternately, if the
            string "count" is passed in, the displayed bars will be colored by
            their y-value (the number of azimuths measurements in that bin).
        Additional keyword arguments are passed on to PatchCollection.

    Returns:
    --------
        A matplotlib PatchCollection
    """
    azimuths = np.asanyarray(azimuths)
    if color_by == 'count':
        z = np.ones_like(azimuths)
        color_by = np.sum
    if ax is None:
        ax = plt.gca()
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(0))
    if bidirectional:
        other = azimuths + 180
        azimuths = np.concatenate([azimuths, other])
        if z is not None:
            z = np.concatenate([z, z])
    # Convert to 0-360, in case negative or >360 azimuths are passed in.
    azimuths[azimuths > 360] -= 360
    azimuths[azimuths < 0] += 360
    counts, edges = np.histogram(azimuths, range=[0, 360], bins=bins)
    if z is not None:
        idx = np.digitize(azimuths, edges)
        z = np.array([color_by(z[idx == i]) for i in range(1, idx.max() + 1)])
        z = np.ma.masked_invalid(z)
    edges = np.radians(edges)
    coll = colored_bar(edges[:-1], counts, z=z, width=np.diff(edges),
                       ax=ax, **kwargs)
    print coll
    return coll


def colored_bar(left, height, z=None, width=0.8, bottom=0, ax=None, **kwargs):
    """A bar plot colored by a scalar sequence."""
    if ax is None:
        ax = plt.gca()
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x, y), w, h))
    coll = PatchCollection(rects, cmap=matplotlib.cm.hsv, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale()
    return coll


if __name__ == '__main__':
    main()