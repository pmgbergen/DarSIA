import darsia
from pathlib import Path
import pandas as pd


class FaciesProps:
    def __init__(
        self,
        facies: darsia.Image,
        porosity: float | dict[int, float] = 1.0,
        permeability: float | dict[int, float] = 1.0,
    ) -> None:
        self.facies = facies
        """Facies label image."""
        if isinstance(porosity, dict):
            self.porosity = darsia.zeros_like(facies, dtype=float)
            """Porosity values for each facies label."""
            for label, value in porosity.items():
                self.porosity[self.facies == label] = value
        else:
            self.porosity = darsia.full_like(facies, fill_value=porosity)
        if isinstance(permeability, dict):
            self.permeability = darsia.zeros_like(facies, dtype=float)
            """Permeability values for each facies label."""
            for label, value in permeability.items():
                self.permeability[self.facies == label] = value
        else:
            self.permeability = darsia.full_like(facies, fill_value=permeability)

    @classmethod
    def load(cls, facies: darsia.Image, path: Path) -> "FaciesProps":
        """Load facies properties from CSV file.

        The CSV file must contain columns 'id', 'porosity', and 'permeability'.

        Args:
            facies (darsia.Image): Facies label image.
            path (Path): Path to CSV file with facies properties.

        Returns:
            FaciesProps: Instance of FaciesProps with loaded properties.

        """
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError("Facies properties file must be .csv or .xlsx format.")
        if (
            "id" in df.columns
            and "porosity" in df.columns
            and "permeability" in df.columns
        ):
            porosity = dict(zip(df["id"].astype(int), df["porosity"]))
            permeability = dict(zip(df["id"].astype(int), df["permeability"]))
            return cls(facies, porosity=porosity, permeability=permeability)
        else:
            raise ValueError(
                "Facies properties file must contain 'id', 'porosity', and 'permeability' columns."
            )
