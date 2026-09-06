"""Array-of-tables section names backing the GUI ``key_list`` widget.

Each ``key_list`` field in a config dataclass declares one of these tuples as its
``key_list_sources`` metadata. The GUI (``SettingsFactory.create_key_list_input``)
unions the entry ``name``s found in ``config_dict[<section>]`` for every listed
section to populate the dropdown, and remembers which section each name came from
(used by type-based ``depends_on`` visibility).

The values match the ``array_key`` metadata already declared on the corresponding
registry dataclasses (``data_registry.py``, ``format_registry.py``,
``roi_registry.py``, ``color_embedding_registry.py``).
"""

REGISTRY_SOURCES = ("data_interval", "data_window", "data_time", "data_path")
FORMAT_SOURCES = ("format",)
ROI_SOURCES = ("roi",)
COLOR_EMBEDDING_SOURCES = ("color_path", "color_range", "color_channel")
