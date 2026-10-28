"""Qt-independent registry for replaceable FluidFlower application pages."""

from __future__ import annotations

import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from darsia.fluidflower.project import FluidFlowerProject


PAGE_ENTRY_POINT_GROUP = "darsia.fluidflower.pages"
SUBPAGE_ENTRY_POINT_GROUP = "darsia.fluidflower.subpages"
BUILTIN_PAGE_PACKAGE = "darsia.fluidflower.pages"
PageFactory = Callable[["FluidFlowerProject"], Any]
SubpageFactory = Callable[[Any], Any]


def _load_factory(target: str) -> PageFactory:
    module_name, separator, attribute = str(target).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "Page factory targets must use the form 'package.module:callable'"
        )
    factory = getattr(import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Page factory {target!r} is not callable")
    return factory


@dataclass(frozen=True, slots=True)
class FluidFlowerPagePlugin:
    """Description of one independently replaceable application page."""

    key: str
    factory: str | PageFactory
    service_modules: tuple[str, ...] = ()
    scrollable: bool = True
    source: str = "builtin"

    def create(self, project: FluidFlowerProject) -> Any:
        factory = (
            _load_factory(self.factory)
            if isinstance(self.factory, str)
            else self.factory
        )
        return factory(project)


@dataclass(frozen=True, slots=True)
class FluidFlowerSubpagePlugin:
    """One tab or nested page contributed to a parent workflow page."""

    parent_key: str
    key: str
    title: str
    factory: str | SubpageFactory
    order: int = 100
    attribute_name: str = ""
    source: str = "builtin"

    def create(self, parent_page: Any) -> Any:
        factory = (
            _load_factory(self.factory)
            if isinstance(self.factory, str)
            else self.factory
        )
        return factory(parent_page)


class FluidFlowerPageRegistry:
    """Runtime page registry shared by the DarSIA shell and external GUIs."""

    def __init__(self, plugins: Iterable[FluidFlowerPagePlugin] = ()) -> None:
        self._plugins: dict[str, FluidFlowerPagePlugin] = {}
        self.discovery_errors: list[str] = []
        for plugin in plugins:
            self.register(plugin)

    def register(
        self,
        plugin: FluidFlowerPagePlugin,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(plugin, FluidFlowerPagePlugin):
            raise TypeError("FluidFlower page registrations require a page plugin")
        key = str(plugin.key or "").strip()
        if not key:
            raise ValueError("FluidFlower page plugins require a non-empty key")
        if key in self._plugins and not replace_existing:
            raise ValueError(f"A FluidFlower page is already registered for {key!r}")
        self._plugins[key] = replace(plugin, key=key)

    def plugin(self, key: str) -> FluidFlowerPagePlugin:
        try:
            return self._plugins[str(key)]
        except KeyError as exc:
            raise KeyError(f"No FluidFlower page is registered for {key!r}") from exc

    def create(self, key: str, project: FluidFlowerProject) -> Any:
        return self.plugin(key).create(project)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._plugins)

    def copy(self) -> FluidFlowerPageRegistry:
        copied = FluidFlowerPageRegistry(self._plugins.values())
        copied.discovery_errors = list(self.discovery_errors)
        return copied

    def discover_package(self, package_name: str, *, strict: bool = True) -> None:
        package = import_module(package_name)
        package_paths = [str(Path(path)) for path in getattr(package, "__path__", ())]
        for module_info in sorted(
            pkgutil.iter_modules(package_paths, f"{package_name}."),
            key=lambda item: item.name,
        ):
            try:
                module = import_module(module_info.name)
                plugin = getattr(module, "PAGE_PLUGIN", None)
                if plugin is not None:
                    self.register(plugin)
            except Exception as exc:
                message = f"{module_info.name}: {type(exc).__name__}: {exc}"
                self.discovery_errors.append(message)
                if strict:
                    raise RuntimeError(
                        f"Could not register built-in FluidFlower page {message}"
                    ) from exc

    def discover_entry_points(
        self,
        *,
        group: str = PAGE_ENTRY_POINT_GROUP,
    ) -> None:
        discovered = entry_points()
        selected = (
            discovered.select(group=group)
            if hasattr(discovered, "select")
            else discovered.get(group, ())
        )
        for entry_point in selected:
            try:
                plugin = entry_point.load()
                if callable(plugin) and not isinstance(plugin, FluidFlowerPagePlugin):
                    plugin = plugin()
                if not isinstance(plugin, FluidFlowerPagePlugin):
                    raise TypeError(
                        "entry point did not provide a FluidFlowerPagePlugin"
                    )
                self.register(
                    replace(plugin, source=f"entry-point:{entry_point.name}"),
                    replace_existing=True,
                )
            except Exception as exc:  # noqa: BLE001
                self.discovery_errors.append(
                    f"{entry_point.name}: {type(exc).__name__}: {exc}"
                )


class FluidFlowerSubpageRegistry:
    """Registry for tabs and nested pages below a workflow module."""

    def __init__(self, plugins: Iterable[FluidFlowerSubpagePlugin] = ()) -> None:
        self._plugins: dict[tuple[str, str], FluidFlowerSubpagePlugin] = {}
        self.discovery_errors: list[str] = []
        for plugin in plugins:
            self.register(plugin)

    def register(
        self,
        plugin: FluidFlowerSubpagePlugin,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(plugin, FluidFlowerSubpagePlugin):
            raise TypeError(
                "FluidFlower subpage registrations require a subpage plugin"
            )
        parent_key = str(plugin.parent_key or "").strip()
        key = str(plugin.key or "").strip()
        if not parent_key or not key:
            raise ValueError("FluidFlower subpages require parent_key and key")
        identity = (parent_key, key)
        if identity in self._plugins and not replace_existing:
            raise ValueError(
                f"A FluidFlower subpage is already registered for {parent_key!r}/{key!r}"
            )
        self._plugins[identity] = replace(
            plugin,
            parent_key=parent_key,
            key=key,
        )

    def plugins_for(self, parent_key: str) -> tuple[FluidFlowerSubpagePlugin, ...]:
        selected = (
            plugin
            for (registered_parent, _key), plugin in self._plugins.items()
            if registered_parent == str(parent_key)
        )
        return tuple(sorted(selected, key=lambda plugin: (plugin.order, plugin.key)))

    def populate_tabs(self, parent_key: str, parent_page: Any, tab_widget: Any) -> None:
        for plugin, widget in self.create_subpages(parent_key, parent_page):
            tab_widget.addTab(widget, plugin.title)

    def populate_pages(self, parent_key: str, parent_page: Any, container: Any) -> None:
        """Populate an ordered page container such as ``QWizard`` without importing Qt."""

        for _plugin, widget in self.create_subpages(parent_key, parent_page):
            container.addPage(widget)

    def create_subpages(
        self,
        parent_key: str,
        parent_page: Any,
    ) -> tuple[tuple[FluidFlowerSubpagePlugin, Any], ...]:
        """Instantiate ordered subpages and expose requested parent attributes."""

        created: list[tuple[FluidFlowerSubpagePlugin, Any]] = []
        for plugin in self.plugins_for(parent_key):
            widget = plugin.create(parent_page)
            if plugin.attribute_name:
                setattr(parent_page, plugin.attribute_name, widget)
            created.append((plugin, widget))
        return tuple(created)

    def copy(self) -> FluidFlowerSubpageRegistry:
        copied = FluidFlowerSubpageRegistry(self._plugins.values())
        copied.discovery_errors = list(self.discovery_errors)
        return copied

    def discover_package(self, package_name: str, *, strict: bool = True) -> None:
        package = import_module(package_name)
        package_paths = [str(Path(path)) for path in getattr(package, "__path__", ())]
        for module_info in sorted(
            pkgutil.iter_modules(package_paths, f"{package_name}."),
            key=lambda item: item.name,
        ):
            try:
                module = import_module(module_info.name)
                for plugin in getattr(module, "SUBPAGE_PLUGINS", ()):
                    self.register(plugin)
            except Exception as exc:
                message = f"{module_info.name}: {type(exc).__name__}: {exc}"
                self.discovery_errors.append(message)
                if strict:
                    raise RuntimeError(
                        f"Could not register built-in FluidFlower subpage {message}"
                    ) from exc

    def discover_entry_points(
        self,
        *,
        group: str = SUBPAGE_ENTRY_POINT_GROUP,
    ) -> None:
        discovered = entry_points()
        selected = (
            discovered.select(group=group)
            if hasattr(discovered, "select")
            else discovered.get(group, ())
        )
        for entry_point in selected:
            try:
                loaded = entry_point.load()
                if callable(loaded) and not isinstance(
                    loaded, FluidFlowerSubpagePlugin
                ):
                    loaded = loaded()
                plugins = (
                    (loaded,)
                    if isinstance(loaded, FluidFlowerSubpagePlugin)
                    else tuple(loaded)
                )
                for plugin in plugins:
                    if not isinstance(plugin, FluidFlowerSubpagePlugin):
                        raise TypeError(
                            "entry point did not provide FluidFlowerSubpagePlugin values"
                        )
                    self.register(
                        replace(plugin, source=f"entry-point:{entry_point.name}"),
                        replace_existing=True,
                    )
            except Exception as exc:  # noqa: BLE001
                self.discovery_errors.append(
                    f"{entry_point.name}: {type(exc).__name__}: {exc}"
                )


def default_page_registry(
    *,
    discover_external: bool = True,
) -> FluidFlowerPageRegistry:
    """Build the standard registry, optionally allowing installed page overrides."""

    from darsia.fluidflower.modules import MODULE_DEFINITIONS

    registry = FluidFlowerPageRegistry()
    registry.discover_package(BUILTIN_PAGE_PACKAGE)
    if discover_external:
        registry.discover_entry_points()
    expected = {"overview", "execution", *(item.key for item in MODULE_DEFINITIONS)}
    missing = sorted(expected.difference(registry.keys()))
    if missing:
        raise RuntimeError(
            "The FluidFlower page registry is incomplete: " + ", ".join(missing)
        )
    return registry


def default_subpage_registry(
    *,
    discover_external: bool = True,
) -> FluidFlowerSubpageRegistry:
    """Build the standard nested-page registry with optional installed overrides."""

    registry = FluidFlowerSubpageRegistry()
    registry.discover_package(BUILTIN_PAGE_PACKAGE)
    if discover_external:
        registry.discover_entry_points()
    return registry


__all__ = [
    "BUILTIN_PAGE_PACKAGE",
    "PAGE_ENTRY_POINT_GROUP",
    "SUBPAGE_ENTRY_POINT_GROUP",
    "FluidFlowerPagePlugin",
    "FluidFlowerPageRegistry",
    "FluidFlowerSubpagePlugin",
    "FluidFlowerSubpageRegistry",
    "PageFactory",
    "SubpageFactory",
    "default_page_registry",
    "default_subpage_registry",
]
