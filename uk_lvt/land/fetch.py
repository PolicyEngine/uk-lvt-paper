"""Download the inputs in ``sources.yaml`` to a git-ignored cache and verify them.

Usage::

    python -m uk_lvt.land.fetch                     # every pinned source
    python -m uk_lvt.land.fetch --only mhclg_lve_2023
    python -m uk_lvt.land.fetch --seed ~/downloads  # reuse local copies by hash
    python -m uk_lvt.land.fetch --offline           # verify the cache, no network

Rules:

* A pinned source is accepted only if its sha256 equals the manifest value.
  A mismatching download is deleted and the run fails; a mismatching file
  already in the cache is left in place and the run fails, so nothing is
  silently replaced.
* ``--seed DIR`` copies a local file into the cache when its sha256 matches a
  pinned source, whatever the local file is called. This is how copies
  downloaded earlier are reused without re-downloading.
* Unpinned sources (``not_fetched``) are skipped unless ``--allow-unpinned``
  is given; then they are downloaded and their sha256 and size are printed so
  they can be pinned in the manifest. ``request`` sources have no URL and are
  always skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import MANIFEST_PATH, REPO_ROOT
from .manifest import Source, load_manifest

USER_AGENT = "uk-lvt-paper land fetch (+https://github.com/PolicyEngine/uk-lvt-paper)"
CHUNK = 1 << 20


class HashMismatch(RuntimeError):
    """A file's sha256 differs from the manifest."""


class MissingInput(RuntimeError):
    """A pinned input is not in the cache and could not be obtained."""


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    action: str  # cached | seeded | downloaded | downloaded_unpinned | skipped
    path: Path | None
    sha256: str | None
    detail: str = ""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def http_download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream ``url`` to ``dest`` (http, https or file URLs)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)


def default_cache_dir(manifest_top: dict | None = None) -> Path:
    env = os.environ.get("UK_LVT_LAND_CACHE")
    if env:
        return Path(env).expanduser()
    rel = (manifest_top or {}).get("default_cache_dir", ".cache/land")
    return REPO_ROOT / rel


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find_seed(src: Source, seed_dirs: Iterable[Path]) -> Path | None:
    """Return a file under ``seed_dirs`` whose sha256 equals the pinned hash.

    Only files whose size equals the pinned size are hashed, so seeding from a
    large directory stays cheap. Directories are scanned one level deep.
    """
    if not src.is_pinned:
        return None
    for d in seed_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for cand in sorted(d.iterdir()):
            if cand.is_file() and cand.stat().st_size == src.bytes:
                if sha256_file(cand) == src.sha256:
                    return cand
    return None


def fetch_source(
    src: Source,
    cache_dir: Path,
    *,
    seed_dirs: Iterable[Path] = (),
    offline: bool = False,
    allow_unpinned: bool = False,
    downloader: Callable[[str, Path], None] = http_download,
) -> FetchResult:
    """Make ``src`` available in ``cache_dir`` and verify it; see module doc."""
    cache_dir = Path(cache_dir)
    if not src.is_downloadable:
        return FetchResult(src.id, "skipped", None, None, f"status={src.status}; no file URL")
    target = cache_dir / src.filename

    if not src.is_pinned:
        if not allow_unpinned:
            return FetchResult(src.id, "skipped", None, None, "not pinned (use --allow-unpinned)")
        if offline:
            return FetchResult(src.id, "skipped", None, None, "not pinned; offline")
        if not target.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=src.filename, suffix=".part")
            os.close(fd)
            try:
                downloader(src.url, Path(tmp))
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        digest = sha256_file(target)
        return FetchResult(
            src.id,
            "downloaded_unpinned",
            target,
            digest,
            f"bytes={target.stat().st_size}; record sha256 and bytes in sources.yaml to pin",
        )

    if target.exists():
        digest = sha256_file(target)
        if digest != src.sha256:
            raise HashMismatch(
                f"{src.id}: cached {target} has sha256 {digest}, manifest says {src.sha256}. "
                "Delete the file to re-fetch, or update the manifest deliberately."
            )
        return FetchResult(src.id, "cached", target, digest)

    seed = find_seed(src, seed_dirs)
    if seed is not None:
        _atomic_copy(seed, target)
        return FetchResult(src.id, "seeded", target, src.sha256, f"from {seed}")

    if offline:
        raise MissingInput(f"{src.id}: {target} is missing and --offline was given")

    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=src.filename, suffix=".part")
    os.close(fd)
    try:
        downloader(src.url, Path(tmp))
        digest = sha256_file(tmp)
        if digest != src.sha256:
            raise HashMismatch(
                f"{src.id}: download from {src.url} has sha256 {digest}, manifest says "
                f"{src.sha256}. The publisher may have replaced the file; the download "
                "was discarded."
            )
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return FetchResult(src.id, "downloaded", target, src.sha256, f"from {src.url}")


def cached_path(source_id: str, cache_dir: Path | None = None, manifest_path: Path = MANIFEST_PATH) -> Path:
    """Path of a verified cached copy of a pinned source (fetch it first)."""
    sources, top = load_manifest(manifest_path)
    src = sources[source_id]
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(top)
    result = fetch_source(src, cache_dir, offline=True)
    assert result.path is not None
    return result.path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="+", metavar="ID", help="source ids to fetch")
    ap.add_argument("--seed", nargs="+", type=Path, default=[], metavar="DIR",
                    help="directories holding local copies to reuse when their sha256 matches")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="also download not_fetched sources and print their hashes")
    args = ap.parse_args(argv)

    sources, top = load_manifest(args.manifest)
    cache_dir = args.cache_dir or default_cache_dir(top)
    ids = args.only or list(sources)
    unknown = [i for i in ids if i not in sources]
    if unknown:
        ap.error(f"unknown source ids: {unknown}")

    failures = 0
    for sid in ids:
        src = sources[sid]
        try:
            res = fetch_source(
                src,
                cache_dir,
                seed_dirs=args.seed,
                offline=args.offline,
                allow_unpinned=args.allow_unpinned,
            )
        except (HashMismatch, MissingInput, OSError) as exc:
            failures += 1
            print(f"FAIL     {sid}: {exc}", file=sys.stderr)
            continue
        sha = (res.sha256 or "")[:12]
        print(f"{res.action:<20} {sid:<45} {sha:<12} {res.detail}")
    print(f"cache: {cache_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
