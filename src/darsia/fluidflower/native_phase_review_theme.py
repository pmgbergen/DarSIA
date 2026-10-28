"""Use the host DarSIA theme for the embedded original review controls."""

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLabel

from darsia.gui.ui.theme import danger_color, muted_text_color, success_color, theme_signal


class NativePhaseReviewTheme(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.sources = {}
        self.applied = {}
        self.updating = False
        self.pending = False
        # No blanket QWidget colours, font sizes or button styling: these are
        # inherited from the same QApplication as the surrounding Dev panels.
        window.setStyleSheet("""
            #pageTitle, #sectionTitle, #overviewTitle { font-weight:600; }
            #workspaceBadge { background:palette(highlight); color:palette(highlighted-text);
                padding:3px 7px; font-weight:600; }
            #blindWorkspaceNotice, #trainingWorkspaceNotice {
                border-left:3px solid palette(highlight); padding:7px 9px; }
            #addQuizRegions, #overviewPanel { border:1px solid palette(mid); }
        """)
        for label in window.findChildren(QLabel):
            # The original review updates some of these styles when the
            # selected frame/answer changes. Keep that semantic information.
            label.installEventFilter(self)
            self.sources[label] = label.styleSheet()
        window.installEventFilter(self)
        QApplication.instance().fontChanged.connect(self.refresh_font)
        theme_signal.theme_changed.connect(self.refresh)
        self.refresh()

    def refresh_font(self, font):
        # Qt stylesheet widgets cache their resolved font; changing the app
        # font alone does not invalidate every descendant of an embedded window.
        style = self.window.styleSheet()
        self.window.setStyleSheet("")
        self.window.setFont(font)
        self.window.setStyleSheet(style)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.StyleChange and watched in self.sources:
            if not self.updating and watched.styleSheet() != self.applied.get(watched):
                self.sources[watched] = watched.styleSheet()
                self.refresh()
        elif watched is self.window and event.type() == QEvent.Type.ApplicationPaletteChange:
            if not self.pending:
                self.pending = True
                QTimer.singleShot(0, self, self.refresh)
        return False

    def refresh(self, *_):
        self.pending = False
        if self.updating:
            return
        palette = QApplication.palette()
        # Warning text must remain readable in Light as well as Dark mode.
        warning = QColor("#986000") if palette.color(QPalette.ColorRole.Window).lightness() > 128 else QColor("#ffd166")
        colours = {
            "#d8dde2": palette.color(QPalette.ColorRole.WindowText),
            "#aeb7bf": muted_text_color(palette),
            "#8d969f": muted_text_color(palette),
            "#ef3e36": danger_color(palette),
            "#ffd166": warning,
            "#9bd4a7": success_color(palette),
        }
        self.updating = True
        try:
            for label, source in self.sources.items():
                styled = source
                for original, colour in colours.items():
                    styled = styled.replace(original, colour.name())
                self.applied[label] = styled
                if label.styleSheet() != styled:
                    label.setStyleSheet(styled)
            self.refresh_rows()
        finally:
            self.updating = False

    def refresh_rows(self):
        # NEW/LABELED text and the original bold NEW font retain the meaning;
        # the view supplies normal, selected and disabled text colours.
        for view in (self.window.new_region_list, self.window.validated_list):
            for index in range(view.count()):
                view.item(index).setData(Qt.ItemDataRole.ForegroundRole, None)
