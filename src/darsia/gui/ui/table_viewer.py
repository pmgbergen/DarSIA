"""Generic read-only table viewer for GUI inspection of tabular data."""

from pathlib import Path
import ast

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

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        title : str
            Window title.
        dataframe : pandas.DataFrame, optional
            DataFrame to display (or None for empty).
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

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
        Pandas.DataFrame with the CSV contents, or empty DataFrame if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_table_reference(value: str, *, loader_key="csv", config=None) -> pd.DataFrame:
    """Read a path or literal ``[workbook, worksheet]`` without evaluating code."""
    text = str(value).strip()
    sheet = None
    if text.startswith(("[", "(")):
        try:
            pair = ast.literal_eval(text)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Enter a file path or [workbook, worksheet]") from error
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0].strip()
                or isinstance(pair[1], bool) or not isinstance(pair[1], (str, int))):
            raise ValueError("Enter a file path or [workbook, worksheet]")
        text, sheet = pair
    from darsia.presets.workflows.config.utils import _config_path, _with_storage_aliases
    path = _config_path(_with_storage_aliases(text), Path(config) if config else None)
    path = Path(_with_storage_aliases(str(path)))
    if path.suffix.casefold() in (".xls", ".xlsx"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if sheet is not None:
        raise ValueError("Only Excel protocol files can name a worksheet")
    return TABLE_LOADERS[loader_key](path)


TABLE_LOADERS = {
    "csv": load_csv_table,
}
