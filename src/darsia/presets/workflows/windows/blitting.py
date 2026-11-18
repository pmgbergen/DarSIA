"""Fast repainting of sliders (blitting).

Without blitting, every slider step re-renders the entire figure, including all images.
The moving parts of the sliders are animated artists instead: they are excluded from the
stored background and drawn on top of it, and a slider step repaints only the region of
that slider.

"""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.widgets import Slider


class SliderBlitter:
    """Repaint sliders of a figure without redrawing the figure.

    Parameters
    ----------
    fig : Figure
        Figure holding the sliders.

    """

    def __init__(self, fig: Figure) -> None:
        self.fig = fig
        self.artists: list = []
        """Moving parts of all sliders (bar, handle, value text)."""
        self.background = None
        """Figure without the moving parts, stored at every full draw."""
        self._extents: dict = {}
        """Last drawn region of each text added with `add_texts`."""
        fig.canvas.mpl_connect("draw_event", self._store_background)

    def add(self, sliders: list[Slider]) -> None:
        """Repaint these sliders by blitting.

        Parameters
        ----------
        sliders : list[Slider]
            Sliders of the figure.

        """
        for slider in sliders:
            for artist in (slider.poly, slider._handle, slider.valtext):
                artist.set_animated(True)
                self.artists.append(artist)
        for slider in sliders:
            slider.drawon = False
            slider.on_changed(lambda val, slider=slider: self.blit(slider))

    def add_texts(self, texts: list) -> None:
        """Repaint these texts by blitting (see `blit_text`).

        Parameters
        ----------
        texts : list[matplotlib.text.Text]
            Texts of the figure which change often (e.g. status messages).

        """
        for text in texts:
            text.set_animated(True)
            self.artists.append(text)
            self._extents[id(text)] = None

    def blit_text(self, text) -> None:
        """Repaint the region of a text after a change (old and new extent).

        Parameters
        ----------
        text : matplotlib.text.Text
            A text added with `add_texts`.

        """
        fig = self.fig
        if self.background is None or not fig.canvas.supports_blit:
            fig.canvas.draw_idle()
            return
        extent = text.get_window_extent(fig.canvas.get_renderer())
        previous = self._extents.get(id(text))
        region = extent if previous is None else Bbox.union([extent, previous])
        self._extents[id(text)] = extent
        fig.canvas.restore_region(self.background)
        for artist in self.artists:
            fig.draw_artist(artist)
        fig.canvas.blit(region.padded(2))

    def _store_background(self, event) -> None:
        fig = self.fig
        self.background = fig.canvas.copy_from_bbox(fig.bbox)
        for artist in self.artists:
            fig.draw_artist(artist)

    def blit(self, slider: Slider) -> None:
        """Repaint the region of a slider.

        Parameters
        ----------
        slider : Slider
            The slider that changed.

        """
        fig = self.fig
        if self.background is None or not fig.canvas.supports_blit:
            fig.canvas.draw_idle()
            return
        handle_radius = 0.5 * slider._handle.get_markersize() * fig.dpi / 72
        region = Bbox.union(
            [slider.ax.bbox, slider.valtext.get_window_extent()]
        ).padded(handle_radius + 1)
        fig.canvas.restore_region(self.background)
        for artist in self.artists:
            fig.draw_artist(artist)
        fig.canvas.blit(region)
