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

import darsia


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


def load_color_path_dir_table(directory: Path) -> pd.DataFrame:
    """Load color paths from a directory and flatten into a DataFrame.

    One row per (label, segment_index) with columns:
    label, segment_index, r, g, b, base_r, base_g, base_b

    Args:
        directory: Path to the directory containing color_path_*.json files.

    Returns:
        pandas.DataFrame with flattened color path data, or empty DataFrame if no files.
    """
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()

    try:
        color_path_map = darsia.LabelColorPathMap.load(directory)
    except Exception:
        return pd.DataFrame()

    if not color_path_map:
        return pd.DataFrame()

    rows = []
    for label, color_path in sorted(color_path_map.items()):
        for segment_idx, color in enumerate(color_path.colors):
            rel_color = (
                color_path.relative_colors[segment_idx]
                if segment_idx < len(color_path.relative_colors)
                else None
            )
            rows.append(
                {
                    "label": label,
                    "segment_index": segment_idx,
                    "r": color[0],
                    "g": color[1],
                    "b": color[2],
                    "rel_r": rel_color[0] if rel_color is not None else 0.0,
                    "rel_g": rel_color[1] if rel_color is not None else 0.0,
                    "rel_b": rel_color[2] if rel_color is not None else 0.0,
                    "base_r": color_path.base_color[0],
                    "base_g": color_path.base_color[1],
                    "base_b": color_path.base_color[2],
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def load_json_dir_table(directory: Path) -> pd.DataFrame:
    """Load color paths from a directory of per-label JSON files.

    Alias for load_color_path_dir_table for consistency with naming conventions.

    Args:
        directory: Path to the directory containing color_path_*.json files.

    Returns:
        pandas.DataFrame with flattened color path data, or empty DataFrame if no files.
    """
    return load_color_path_dir_table(directory)


TABLE_LOADERS = {
    "csv": load_csv_table,
    "color_path_dir": load_color_path_dir_table,
    "json_dir": load_json_dir_table,
}
