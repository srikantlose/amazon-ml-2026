"""Import the real src.lookalike without pulling in torch (src.blocking) and without writing .pyc files."""
import sys
import types

sys.dont_write_bytecode = True
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src  # noqa: E402  (empty package)

_b = types.ModuleType("src.blocking")
_b.pruned_path = lambda cfg, split: None
_f = types.ModuleType("src.features")
_f.features_dir = lambda *a, **k: None
_f.write_parts = lambda *a, **k: None
sys.modules.setdefault("src.blocking", _b)
sys.modules.setdefault("src.features", _f)

import src.lookalike as la  # noqa: E402

CACHE = ROOT + r"\data\cache"
