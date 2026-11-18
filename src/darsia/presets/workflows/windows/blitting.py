"""Fast repainting of sliders, texts and panels (blitting).

Without blitting, every slider step re-renders the entire figure, including all images.
The moving parts of the sliders are animated artists instead: they are excluded from the
stored background and drawn on top of it, and a slider step repaints only the region of
that slider. Status texts are repainted the same way, and changed panels (axes) can be
repainted alone (`redraw`).

"""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, IdentityTransform
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
        self._panels: dict = {}
        """Last drawn region of each panel added with `add_panels`."""
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
        renderer = fig.canvas.get_renderer()
        for ax in self._panels:
            self._panels[ax] = ax.get_tightbbox(renderer)
        for artist in self.artists:
            fig.draw_artist(artist)

    def add_panels(self, axes: list) -> None:
        """Axes which `redraw` may repaint alone.

        Parameters
        ----------
        axes : list[matplotlib.axes.Axes]
            Panels of the figure (not overlapping each other).

        """
        for ax in axes:
            self._panels.setdefault(ax, None)

    def redraw(self, axes: list, artists: list | None = None) -> None:
        """Repaint some panels instead of the whole figure.

        The region of each panel (its box and what it drew before and now, including
        titles and tick labels) is cleared, then the panels and further artists are
        drawn anew; the stored background is updated, then the animated artists are
        drawn on top. Without blitting support (or before the first draw), the whole
        figure is redrawn.

        Parameters
        ----------
        axes : list[matplotlib.axes.Axes]
            Changed panels, drawn in this order.
        artists : list, optional
            Further changed artists (e.g. figure texts) inside the regions of the
            panels, drawn after them.

        """
        fig = self.fig
        canvas = fig.canvas
        if self.background is None or not canvas.supports_blit:
            canvas.draw_idle()
            return
        renderer = canvas.get_renderer()
        canvas.restore_region(self.background)

        # Extent of the changed panels now (measured once; measuring texts is slow),
        # and of all panels as drawn before
        now = {ax: ax.get_tightbbox(renderer) for ax in axes}

        def region_of(ax) -> Bbox:
            boxes = [ax.bbox, now.get(ax), self._panels.get(ax)]
            return Bbox.union([box for box in boxes if box is not None]).padded(2)

        # Unchanged panels whose drawing reaches into a cleared region (e.g. a long
        # text) are drawn as well
        axes = list(axes)
        regions = [region_of(ax) for ax in axes]
        added = True
        while added:
            added = False
            for ax, drawn in self._panels.items():
                if ax in axes or drawn is None:
                    continue
                if any(drawn.overlaps(region) for region in regions):
                    axes.append(ax)
                    regions.append(region_of(ax))
                    added = True

        for region in regions:
            clear = Rectangle(
                (region.x0, region.y0),
                region.width,
                region.height,
                transform=IdentityTransform(),
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            clear.set_figure(fig)
            clear.draw(renderer)
        for ax in axes:
            ax.draw(renderer)
            if ax in self._panels and ax in now:
                self._panels[ax] = now[ax]
        for artist in artists or []:
            artist.draw(renderer)
        self.background = canvas.copy_from_bbox(fig.bbox)
        for artist in self.artists:
            fig.draw_artist(artist)
        for region in regions:
            canvas.blit(region)

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
