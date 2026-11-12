from pathlib import Path
from typing import Optional, Tuple

from .base import BatchPotential
from .mace import MACEBatchBackend
from .mock import MockBatchBackend
from .sevennet import SevenNetBatchBackend
from .chgnet import CHGNetBatchBackend
from .matris import DEFAULT_MATRIS_MODEL, MatRISBatchBackend
from .eqv3 import DEFAULT_EQV3_MODEL, EqV3BatchBackend


SUPPORTED_BACKENDS = ("mace", "mock", "matris", "chgnet", "eqv3")
UNIMPLEMENTED_BACKENDS = ("sevennet", "matgl")

#: Cache directory conventions of the foundation-model loaders.
MACE_CACHE_DIR = Path.home() / ".cache" / "mace"
MATRIS_CACHE_DIR = Path.home() / ".cache" / "matris"
MATRIS_MODEL_NAMES = (
    "matris_10m_omat",
    "matris_10m_oam",
    "matris_10m_mp",
    "matris_4m_matpes_pbev1",
    "matris_4m_matpes_pbev2",
    "matris_4m_matpes_r2scanv1",
    "matris_4m_matpes_r2scanv2",
)

#: MACE family aliases -> loader hint ("off" = MACE-OFF23, "mp" = MACE-MP).
_MACE_FAMILY_LOADERS = {
    "mace": "off",  # legacy alias, identical to "mace_off:small"
    "mace_off": "off",
    "mace_mp": "mp",
}


def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """Split a model identifier ``family[:spec]`` into family and spec.

    The family is case-insensitive; an empty spec after ':' yields None.
    """
    text = (model_id or "").strip()
    if not text:
        raise ValueError("Model identifier must not be empty")
    if ":" in text:
        family, spec = text.split(":", 1)
        family = family.strip().lower()
        spec = spec.strip() or None
        if not family:
            raise ValueError(f"Invalid model identifier '{model_id}': empty family before ':'")
        return family, spec
    return text.lower(), None


def resolve_checkpoint(spec: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a checkpoint spec to an existing local file, or return None.

    Search order:
      1. ``spec`` itself as a filesystem path (``~`` expanded).
      2. ``<cache_dir>/<spec>`` (default ``~/.cache/mace``), matching the
         caching convention of the MACE foundation-model loaders.

    Returning ``None`` means the spec is treated as a family-internal model
    name (e.g. ``"medium-mpa-0"``, ``"7net-0"``) and is left to the family's
    own loader, which may resolve or download it.
    """
    if not spec:
        return None
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate
    base = Path(cache_dir) if cache_dir is not None else MACE_CACHE_DIR
    cached = base / spec
    if cached.is_file():
        return cached
    return None


def _is_known_mace_name(loader: str, spec: str) -> bool:
    """Whether spec is a built-in model name of the given MACE loader family."""
    try:
        from mace.calculators.foundations_models import mace_mp_urls, mace_off_urls
    except ImportError:
        return False
    urls = mace_off_urls if loader == "off" else mace_mp_urls
    return spec in urls


def resolve_matris_checkpoint(
    spec: str, cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """Resolve a MatRIS checkpoint path or a filename in its model cache."""
    if not spec:
        return None
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate
    base = Path(cache_dir) if cache_dir is not None else MATRIS_CACHE_DIR
    cached = base / spec
    return cached if cached.is_file() else None


def validate_model_id(model_id: str) -> Optional[Path]:
    """Validate a model identifier without instantiating any backend.

    Returns the resolved local checkpoint path (None when the family resolves
    its checkpoint internally, e.g. built-in names or no spec).

    Raises:
        ValueError: unknown family or malformed identifier.
        NotImplementedError: placeholder family (sevennet/matgl).
        FileNotFoundError: an eqV3 checkpoint cannot be resolved locally.
    """
    family, spec = parse_model_id(model_id)
    if family in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{family}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    if family not in _MACE_FAMILY_LOADERS and family not in ("mock", "matris", "chgnet", "eqv3"):
        raise ValueError(
            f"Unknown potential backend: '{family}'. Supported: {list(SUPPORTED_BACKENDS)}"
        )
    if family == "eqv3":
        checkpoint = Path(spec or DEFAULT_EQV3_MODEL).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"eqV3 checkpoint was not found: '{checkpoint}'")
        return checkpoint
    if family in ("mock", "chgnet") or spec is None:
        return None
    if family == "matris":
        resolved = resolve_matris_checkpoint(spec)
        if resolved is not None:
            return resolved
        if spec.lower() in MATRIS_MODEL_NAMES or spec.startswith(("http://", "https://")):
            return None
        raise FileNotFoundError(
            f"MatRIS checkpoint '{spec}' was not found as a local path "
            f"('{Path(spec).expanduser()}') nor in the MatRIS cache "
            f"('{MATRIS_CACHE_DIR / spec}'), and it is not a built-in model name."
        )
    resolved = resolve_checkpoint(spec)
    if resolved is not None:
        return resolved
    if _is_known_mace_name(_MACE_FAMILY_LOADERS[family], spec):
        return None
    raise FileNotFoundError(
        f"Checkpoint spec '{spec}' for family '{family}' was not found as a local path "
        f"('{Path(spec).expanduser()}') nor in the MACE cache ('{MACE_CACHE_DIR / spec}'), "
        f"and it is not a built-in {family} model name."
    )


def create_backend(model_id: str = "mace", **kwargs) -> BatchPotential:
    """Factory creating potential backend by model identifier.

    Identifier grammar: ``family[:spec]``. Supported families:

    - ``mace``            MACE-OFF23 small (legacy default, = ``mace_off:small``)
    - ``mace_off[:spec]`` MACE-OFF23 family (organic elements only); ``spec`` is
                          ``small``/``medium``/``large``, a cached model file name
                          (resolved in ``~/.cache/mace``), or a local path
    - ``mace_mp[:spec]``  MACE-MP family (89 elements incl. Li); ``spec`` is an
                          official name (e.g. ``medium-mpa-0``), a cached model
                          file name, or a local path (default: mace-mpa-0-medium)
    - ``matris[:spec]``   MatRIS family; ``spec`` is an official model name, a
                          cached filename in ``~/.cache/matris``, a local
                          checkpoint path, or an HTTP(S) checkpoint URL
    - ``chgnet[:spec]``   CHGNet family; ``spec`` is a released model name
    - ``eqv3[:spec]``     eqV3 family; ``spec`` is a local FairChem checkpoint
    - ``mock``            Mock backend for tests

    Raises:
        NotImplementedError: If the requested family is a known placeholder
            under active development.
        ValueError: If the requested family is unknown.
    """
    family, spec = parse_model_id(model_id)

    if family in _MACE_FAMILY_LOADERS:
        loader_hint = kwargs.pop("loader", _MACE_FAMILY_LOADERS[family])
        model_value = kwargs.pop("model", None)
        if model_value is None and spec is not None:
            checkpoint = resolve_checkpoint(spec)
            # Bare built-in names (e.g. "medium-mpa-0") are forwarded to the
            # MACE loader which resolves/downloads them; resolved file paths
            # are used as-is.
            model_value = str(checkpoint) if checkpoint is not None else spec
        return MACEBatchBackend(model=model_value, loader=loader_hint, **kwargs)

    if family == "matris":
        model_value = kwargs.pop("model", None)
        model_name = kwargs.pop("model_name", None)
        if model_value is not None and model_name is not None:
            raise ValueError("Pass only one of 'model' and 'model_name'.")
        if model_value is None:
            model_value = model_name
        if model_value is None and spec is not None:
            checkpoint = resolve_matris_checkpoint(spec)
            model_value = str(checkpoint) if checkpoint is not None else spec
        if model_value is None:
            model_value = DEFAULT_MATRIS_MODEL
        return MatRISBatchBackend(model=model_value, **kwargs)

    if family == "chgnet":
        model_name = kwargs.pop("model_name", spec or "0.3.0")
        return CHGNetBatchBackend(model_name=model_name, **kwargs)

    if family == "eqv3":
        checkpoint = kwargs.pop("checkpoint_path", spec or DEFAULT_EQV3_MODEL)
        return EqV3BatchBackend(checkpoint_path=checkpoint, **kwargs)

    if family == "mock":
        return MockBatchBackend(**kwargs)

    if family in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{family}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    raise ValueError(
        f"Unknown potential backend: '{family}'. Supported: {list(SUPPORTED_BACKENDS)}"
    )


__all__ = [
    "BatchPotential",
    "MACEBatchBackend",
    "MockBatchBackend",
    "SevenNetBatchBackend",
    "CHGNetBatchBackend",
    "MatRISBatchBackend",
    "EqV3BatchBackend",
    "SUPPORTED_BACKENDS",
    "UNIMPLEMENTED_BACKENDS",
    "MACE_CACHE_DIR",
    "MATRIS_CACHE_DIR",
    "MATRIS_MODEL_NAMES",
    "DEFAULT_MATRIS_MODEL",
    "DEFAULT_EQV3_MODEL",
    "parse_model_id",
    "resolve_checkpoint",
    "resolve_matris_checkpoint",
    "validate_model_id",
    "create_backend",
]
