"""Generic read-only table viewer for GUI inspection of tabular data."""

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class TableViewerDialog(QDialog):
    """Read-only table viewer dialog for displaying tabular data."""

    def __init__(self, parent=None, title="Data Viewer", dataframe=None):
        """Initialize the table viewer dialog.

        Args:
            parent: Parent widget.
            title: Window title.
            dataframe: pandas.DataFrame to display (or None for empty).
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(self)

        if dataframe is None or dataframe.empty:
            label = QLabel("No data available.")
            layout.addWidget(label)
        else:
            table = QTableWidget()
            table.setRowCount(len(dataframe))
            table.setColumnCount(len(dataframe.columns))
            table.setHorizontalHeaderLabels(dataframe.columns)

            for row_idx, row in dataframe.iterrows():
                for col_idx, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row_idx, col_idx, item)

            table.resizeColumnsToContents()
            layout.addWidget(table)

        self.setLayout(layout)


def load_csv_table(path: Path) -> pd.DataFrame:
    """Load a CSV file as a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        pandas.DataFrame with the CSV contents, or empty DataFrame if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


TABLE_LOADERS = {
    "csv": load_csv_table,
}
