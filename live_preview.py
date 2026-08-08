import matplotlib.pyplot as plt

THREAD_ALPHA = 0.8
THREAD_WIDTH = 0.5


class LivePreview:
    """A live-updating plot of nails and the threads strung between them."""

    def __init__(self, nails, title):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.scatter(nails[:, 0], nails[:, 1], color="lightgray", s=2, zorder=2)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(title)

        self._drawn_line, = self.ax.plot([], [], color="black", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)
        self._next_line, = self.ax.plot([], [], color="red", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)
        self._drawn_x = [nails[0, 0]]
        self._drawn_y = [nails[0, 1]]

        plt.show(block=False)

    def add_point(self, x, y, refresh=True):
        """Extend the drawn (black) thread history to include (x, y)."""
        self._drawn_x.append(x)
        self._drawn_y.append(y)
        self._drawn_line.set_data(self._drawn_x, self._drawn_y)
        if refresh:
            self.refresh()

    def show_next(self, x0, y0, x1, y1):
        """Preview the thread about to be drawn, in red."""
        self._next_line.set_data([x0, x1], [y0, y1])
        self.refresh()

    def clear_next(self):
        self._next_line.set_data([], [])

    def set_title(self, title):
        self.ax.set_title(title)

    def refresh(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finish(self):
        plt.ioff()
        plt.show()
