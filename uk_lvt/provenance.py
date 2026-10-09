"""Dataset pin and version stamp written into every results JSON.

Every generator in this repository runs on one Enhanced FRS file. This module
holds that pin in one place, lets a reproducer point the generators at a local
copy of the same file, and builds the ``provenance`` block that each results
JSON carries: the policyengine-uk and policyengine-core versions, the numpy
and pandas versions, the dataset source and the SHA-256 of the dataset file
that was actually read.

Nothing here imports policyengine at module level, so the committed-results
tests can import it without the ``simulation`` extra installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
from pathlib import Path

# The file the paper's results were generated from: ``enhanced_frs_2023_24.h5``
# at revision ``1.56.14`` of the private Hugging Face repository. The blob at
# that revision was uploaded with data release 1.55.10 and is not listed in
# the 1.56.14 release manifest; the revision tag and the hash below identify
# it exactly.
PINNED_DATASET = (
    "hf://policyengine/policyengine-uk-data-private/enhanced_frs_2023_24.h5@1.56.14"
)
PINNED_DATASET_SHA256 = (
    "584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"
)

# Environment variable that points every generator at another copy of the
# dataset (for example a local download of the pinned file).
DATASET_ENV_VAR = "UK_LVT_DATASET"

_PACKAGES = ("policyengine-uk", "policyengine-core", "numpy", "pandas")


def dataset_source() -> str:
    """Dataset the generators should read: ``$UK_LVT_DATASET`` or the pin."""
    return os.environ.get(DATASET_ENV_VAR) or PINNED_DATASET


def resolve_dataset_path(dataset: str) -> Path:
    """Return a local path for ``dataset``, downloading ``hf://`` sources.

    ``hf://owner/repo/path/to/file[@revision]`` URLs are split with
    policyengine-core's ``parse_hf_url``, which keeps the owner and any
    directories in the file path, and fetched through policyengine-core's
    Hugging Face helper, which uses the same local cache the simulation reads
    from. For a file at the top of the repository, such as the pin, this
    requests the same repository, file and revision as the simulation's own
    download. policyengine-uk 2.89.4 takes the last three path segments as
    owner, repository and file, so for a file in a subdirectory the simulation
    itself requests the wrong repository.
    """
    if dataset.startswith("hf://"):
        from policyengine_core.tools.hugging_face import (
            download_huggingface_dataset,
            parse_hf_url,
        )

        owner, repo, filename, version = parse_hf_url(dataset)
        return Path(
            download_huggingface_dataset(
                repo=f"{owner}/{repo}", repo_filename=filename, version=version
            )
        )
    if "://" in dataset:
        raise ValueError(f"Unsupported dataset source for hashing: {dataset}")
    path = Path(dataset).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that determine the numbers."""
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def build_provenance(dataset: str) -> dict:
    """Provenance block for a results JSON generated from ``dataset``.

    The block deliberately carries no timestamp or host information, so two
    runs on the same pinned inputs write byte-identical files.
    """
    sha256 = sha256_file(resolve_dataset_path(dataset))
    return {
        "packages": package_versions(),
        "dataset": dataset,
        "dataset_sha256": sha256,
        "dataset_matches_pin": sha256 == PINNED_DATASET_SHA256,
    }
