from src.quantum_algorithms.vqe.distance_finder import box_size


def setup_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim([-box_size, box_size])
    ax.set_ylim([-box_size, box_size])
    ax.set_zlim([-box_size, box_size])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def plot_point(coords, ax, symbol="o", size=None):
    size = size or 20
    ax.scatter(*coords, marker=symbol, s=size, c="b")
    x, y, z = tuple(coords)
    markersize = size / 10 if size is not None else None
    ax.plot(x, z, f'r{symbol}', markersize=markersize, zdir='y', zs=box_size, markerfacecolor="none")
    ax.plot(y, z, f'g{symbol}', markersize=markersize, zdir='x', zs=-box_size, markerfacecolor="none")
    ax.plot(x, y, f'k{symbol}', markersize=markersize, zdir='z', zs=-box_size, markerfacecolor="none")
    plt.draw()
    plt.pause(0.05)
