"""CBraMod and EEGSimpleConv experiments on SHU-MI.

The package root intentionally stays lightweight. Import concrete public APIs from
``cbramod_experiments.datasets``, ``cbramod_experiments.models`` or
``cbramod_experiments.utils``. Avoiding eager subpackage imports here prevents
partially-initialized modules and circular-import failures.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
