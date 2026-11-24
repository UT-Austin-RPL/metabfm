import os as _os

from metabfm.config_registry import ConfigRegistry

configs = ConfigRegistry(_os.path.join(_os.path.dirname(__file__), "configs"))
