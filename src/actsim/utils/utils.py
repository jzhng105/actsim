import yaml
import os
import warnings
from typing import Any, Dict
import time


def _default_config_path() -> str:
    """Locate the packaged config.yaml.

    Prefers importlib.resources, which is the supported API on Python 3.9+ and
    the only one that works on setuptools 81+, where pkg_resources was removed.
    Falls back to pkg_resources for older interpreters that lack files().
    """
    try:
        from importlib.resources import files

        return str(files("actsim").joinpath("config.yaml"))
    except (ImportError, AttributeError, ModuleNotFoundError):
        import pkg_resources  # type: ignore[import-untyped]

        return pkg_resources.resource_filename("actsim", "config.yaml")


class Config:
    def __init__(self, file_path: str = None):
        """
        Initialize Config with a YAML file path.
        
        Args:
            file_path: Path to the YAML config file. If None, uses the default config.yaml
                      from the package installation.
        """
        if file_path is None:
            # Use the default config file from the package
            try:
                file_path = _default_config_path()
            except Exception as e:
                warnings.warn(f"Could not find default config file: {e}")
                file_path = "config.yaml"  # Fallback to local file
        
        self.__dict__['_file_path'] = file_path
        self.__dict__['_data'] = self._read_yaml()

    def _read_yaml(self) -> Dict[str, Any]:
        """Reads the YAML file and returns its contents as a dictionary."""
        if not os.path.exists(self._file_path):
            warnings.warn("File path doesn't exist", UserWarning)
            return {}
        with open(self._file_path, 'r') as file:
            content = yaml.safe_load(file)

            if content is None:
                warnings.warn("The YAML file is empty or contains no data.", UserWarning)
                return {}

            return content

    def save(self) -> None:
        """Writes the current data to the YAML file."""
        with open(self._file_path, 'w') as file:
            yaml.dump(self._data, file, default_flow_style=False)

    def __getattr__(self, key: str) -> Any:
        """Gets a value using attribute-style access."""
        return self._data.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Sets a value using attribute-style access and saves the file."""
        self._data[key] = value
        self.save()

    def __delattr__(self, key: str) -> None:
        """Deletes a key using attribute-style access and saves the file."""
        if key in self._data:
            del self._data[key]
            self.save()

    def has_key(self, key: str) -> bool:
        """Checks if a specific key exists in the YAML file data."""
        return key in self._data

    def update(self, updates: Dict[str, Any]) -> None:
        """Updates multiple values in the YAML file data and saves the file."""
        self._data.update(updates)
        self.save()

    def clear(self) -> None:
        """Clears all data in the YAML file."""
        self._data = {}
        self.save()

    def reload(self) -> None:
        """Reloads the data from the YAML file."""
        self._data = self._read_yaml()

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper