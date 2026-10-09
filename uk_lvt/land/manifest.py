"""Load and validate the input manifest (``sources.yaml``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import MANIFEST_PATH

STATUSES = ("pinned", "not_fetched", "request")
REQUIRED_KEYS = (
    "id",
    "title",
    "publisher",
    "landing_page",
    "url",
    "filename",
    "edition",
    "period",
    "licence",
    "status",
    "retrieved_utc",
    "bytes",
    "sha256",
    "remote_check",
    "used_for",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9_]+$")


class ManifestError(ValueError):
    """The manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    publisher: str
    landing_page: str | None
    url: str | None
    filename: str | None
    edition: str | None
    period: str | None
    licence: str
    status: str
    retrieved_utc: str | None
    bytes: int | None
    sha256: str | None
    remote_check: str
    used_for: tuple[str, ...]
    notes: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_pinned(self) -> bool:
        return self.status == "pinned"

    @property
    def is_downloadable(self) -> bool:
        return self.url is not None and self.filename is not None


def _validate_entry(raw: dict) -> Source:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ManifestError(f"{raw.get('id', '?')}: missing keys {missing}")
    sid = raw["id"]
    if not isinstance(sid, str) or not _ID.match(sid):
        raise ManifestError(f"bad id {sid!r}")
    status = raw["status"]
    if status not in STATUSES:
        raise ManifestError(f"{sid}: status {status!r} not in {STATUSES}")
    sha = raw["sha256"]
    if sha is not None and (not isinstance(sha, str) or not _SHA256.match(sha)):
        raise ManifestError(f"{sid}: sha256 must be 64 lowercase hex chars or null")
    if status == "pinned":
        for key in ("url", "filename", "sha256", "bytes", "retrieved_utc"):
            if raw[key] in (None, ""):
                raise ManifestError(f"{sid}: pinned source needs {key}")
        if not isinstance(raw["bytes"], int) or raw["bytes"] <= 0:
            raise ManifestError(f"{sid}: bytes must be a positive integer")
    else:
        if sha is not None or raw["bytes"] is not None:
            raise ManifestError(
                f"{sid}: only pinned sources may carry sha256/bytes "
                "(pin them by changing status to 'pinned')"
            )
    if status == "request" and raw["url"] is not None:
        raise ManifestError(f"{sid}: a 'request' source has no public url")
    if raw["filename"] is not None and ("/" in raw["filename"] or raw["filename"].startswith(".")):
        raise ManifestError(f"{sid}: filename must be a plain file name")
    used_for = raw["used_for"]
    if not isinstance(used_for, list) or not used_for:
        raise ManifestError(f"{sid}: used_for must be a non-empty list")
    known = set(REQUIRED_KEYS) | {"notes"}
    return Source(
        id=sid,
        title=raw["title"],
        publisher=raw["publisher"],
        landing_page=raw["landing_page"],
        url=raw["url"],
        filename=raw["filename"],
        edition=raw["edition"],
        period=raw["period"],
        licence=raw["licence"],
        status=status,
        retrieved_utc=raw["retrieved_utc"],
        bytes=raw["bytes"],
        sha256=sha,
        remote_check=raw["remote_check"],
        used_for=tuple(str(u) for u in used_for),
        notes=raw.get("notes"),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def parse_manifest(data: dict) -> dict[str, Source]:
    """Validate a parsed manifest mapping and return sources keyed by id."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ManifestError("manifest must be a mapping with schema_version: 1")
    entries = data.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest needs a non-empty 'sources' list")
    out: dict[str, Source] = {}
    filenames: dict[str, str] = {}
    for raw in entries:
        src = _validate_entry(raw)
        if src.id in out:
            raise ManifestError(f"duplicate id {src.id}")
        if src.filename is not None:
            if src.filename in filenames:
                raise ManifestError(
                    f"{src.id} and {filenames[src.filename]} share filename {src.filename}"
                )
            filenames[src.filename] = src.id
        out[src.id] = src
    return out


def load_manifest(path: Path | str = MANIFEST_PATH) -> tuple[dict[str, Source], dict]:
    """Return ``(sources_by_id, top_level_mapping)`` for a manifest file."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    return parse_manifest(data), data
