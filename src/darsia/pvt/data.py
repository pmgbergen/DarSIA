"""Material data of the supported fluids (moved from `CO2MassAnalysis`).

Units: pressure in bar, temperature in degrees Celsius, density and solubility in kg/m3.
The CO2 densities are NIST data (isothermal, at 20 and 23 degrees Celsius); the water
densities are the default values of the lab setup at those temperatures.
"""

# Temperatures (degrees Celsius) at which all tables are given.
TEMPERATURES = (20.0, 23.0)
"""Temperatures of the tables in degrees Celsius."""

WATER_DENSITY = (998.21, 997.62)
"""Water density in kg/m3 at 20 and 23 degrees Celsius (independent of pressure)."""

CO2_DENSITY_PRESSURES = tuple(0.90 + 0.01 * i for i in range(61))
"""Pressures in bar of the NIST CO2 density tables (0.90 to 1.50 bar)."""

CO2_DENSITY_20 = (
    1.6328,
    1.651,
    1.6692,
    1.6875,
    1.7057,
    1.7239,
    1.7422,
    1.7604,
    1.7787,
    1.7969,
    1.8152,
    1.8334,
    1.8517,
    1.8699,
    1.8882,
    1.9064,
    1.9247,
    1.9429,
    1.9612,
    1.9795,
    1.9977,
    2.016,
    2.0343,
    2.0526,
    2.0708,
    2.0891,
    2.1074,
    2.1257,
    2.1439,
    2.1622,
    2.1805,
    2.1988,
    2.2171,
    2.2354,
    2.2537,
    2.272,
    2.2903,
    2.3086,
    2.3269,
    2.3452,
    2.3635,
    2.3818,
    2.4001,
    2.4186,
    2.4367,
    2.455,
    2.4734,
    2.4917,
    2.51,
    2.5283,
    2.5467,
    2.565,
    2.5833,
    2.6016,
    2.62,
    2.6383,
    2.6566,
    2.675,
    2.6933,
    2.7117,
    2.73,
)
"""CO2 density in kg/m3 at 20 degrees Celsius."""

CO2_DENSITY_23 = (
    1.616,
    1.634,
    1.6521,
    1.6701,
    1.6882,
    1.7062,
    1.7242,
    1.7423,
    1.7604,
    1.7784,
    1.7965,
    1.8145,
    1.8326,
    1.8506,
    1.8687,
    1.8868,
    1.9048,
    1.9229,
    1.941,
    1.959,
    1.9771,
    1.9952,
    2.0133,
    2.0314,
    2.0494,
    2.0675,
    2.0856,
    2.1037,
    2.1218,
    2.1399,
    2.158,
    2.1761,
    2.1942,
    2.2123,
    2.2304,
    2.2485,
    2.2666,
    2.2847,
    2.3028,
    2.3209,
    2.339,
    2.3571,
    2.3752,
    2.3934,
    2.4115,
    2.4296,
    2.4477,
    2.4658,
    2.484,
    2.5021,
    2.5202,
    2.5384,
    2.5565,
    2.5746,
    2.5928,
    2.6109,
    2.6291,
    2.6472,
    2.6653,
    2.6835,
    2.7016,
)
"""CO2 density in kg/m3 at 23 degrees Celsius."""

CO2_SOLUBILITY_PRESSURES = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
"""Pressures in bar of the CO2 solubility tables."""

CO2_SOLUBILITY_20 = (1.53, 1.7, 1.87, 2.04, 2.21, 2.37, 2.54)
"""Solubility of CO2 in water in kg/m3 at 20 degrees Celsius."""

CO2_SOLUBILITY_23 = (1.3, 1.45, 1.6, 1.75, 1.9, 2.05, 2.18)
"""Solubility of CO2 in water in kg/m3 at 23 degrees Celsius."""
