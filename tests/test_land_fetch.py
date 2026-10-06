"""Tests for the input manifest and the hash-verifying fetcher (no network)."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from uk_lvt.land import MANIFEST_PATH
from uk_lvt.land import fetch as F
from uk_lvt.land.manifest import ManifestError, load_manifest, parse_manifest

# Every input named in the work plan's deliverable-1 input list.
PLAN_INPUT_IDS = {
    "mhclg_lve_2023",
    "ukhpi_average_prices_2026_07",
    "hmlr_price_paid_complete",
    "epc_d3_domestic",
    "voa_ctsop_2026_summary",
    "ons_nbs_2025",
    "ons_land_under_dwellings_latest",
    "mhclg_live_table_125",
    "mhclg_live_table_126",
    "ons_housing_affordability_2025",
    "mhclg_council_tax_levels_2026_27_tables_1_9",
    "wales_council_tax_levels_2026_27",
    "scotland_council_tax_by_band_2026_27",
    "tpa_lvt_model_cells",
    "progress_and_property_la_aggregates",
}


# --------------------------------------------------------------------------
# The real manifest
# --------------------------------------------------------------------------

def test_real_manifest_validates_and_lists_every_plan_input():
    sources, top = load_manifest()
    assert PLAN_INPUT_IDS <= set(sources)
    assert top["default_cache_dir"] == ".cache/land"
    for s in sources.values():
        assert s.licence and s.publisher and s.used_for
        if s.is_pinned:
            assert s.url.startswith("https://") and s.retrieved_utc.endswith("Z")
        if s.status == "request":
            assert s.url is None


def test_cache_dir_is_git_ignored():
    gitignore = (MANIFEST_PATH.parents[2] / ".gitignore").read_text().splitlines()
    assert ".cache/" in gitignore


# --------------------------------------------------------------------------
# Manifest validation
# --------------------------------------------------------------------------

def _entry(**over):
    e = {
        "id": "x", "title": "t", "publisher": "p", "landing_page": None,
        "url": "https://example.invalid/x.csv", "filename": "x.csv", "edition": "e",
        "period": "p", "licence": "OGL-UK-3.0", "status": "pinned",
        "retrieved_utc": "2026-10-06T00:00:00Z", "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(), "remote_check": "none",
        "used_for": ["test"],
    }
    e.update(over)
    return e


def _manifest(*entries):
    return {"schema_version": 1, "sources": list(entries)}


@pytest.mark.parametrize(
    "bad",
    [
        _manifest(_entry(), _entry()),                                   # duplicate id
        _manifest(_entry(sha256=None)),                                  # pinned without hash
        _manifest(_entry(sha256="ABC")),                                 # malformed hash
        _manifest(_entry(status="not_fetched")),                         # unpinned carrying a hash
        _manifest(_entry(status="request", sha256=None, bytes=None)),    # request with a url
        _manifest(_entry(status="bogus")),
        _manifest(_entry(filename="../x.csv")),
        _manifest(_entry(id="a"), _entry(id="b")),                       # shared filename
        {"schema_version": 2, "sources": [_entry()]},
    ],
)
def test_manifest_rejects_inconsistent_entries(bad):
    with pytest.raises(ManifestError):
        parse_manifest(bad)


# --------------------------------------------------------------------------
# fetch_source
# --------------------------------------------------------------------------

PAYLOAD = b"col\n1\n"


def _source(payload=PAYLOAD, **over):
    entry = _entry(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), **over)
    return parse_manifest(_manifest(entry))["x"]


def _writer(payload):
    calls = []

    def download(url, dest):
        calls.append(url)
        Path(dest).write_bytes(payload)

    download.calls = calls
    return download


def test_download_accepted_when_hash_matches(tmp_path):
    dl = _writer(PAYLOAD)
    res = F.fetch_source(_source(), tmp_path, downloader=dl)
    assert res.action == "downloaded"
    assert (tmp_path / "x.csv").read_bytes() == PAYLOAD
    assert dl.calls == ["https://example.invalid/x.csv"]
    # Second call uses the verified cache, no download.
    res2 = F.fetch_source(_source(), tmp_path, downloader=dl)
    assert res2.action == "cached" and len(dl.calls) == 1


def test_download_discarded_when_hash_differs(tmp_path):
    with pytest.raises(F.HashMismatch):
        F.fetch_source(_source(), tmp_path, downloader=_writer(b"tampered\n"))
    assert list(tmp_path.iterdir()) == []  # neither the target nor a .part file remains


def test_mismatching_cached_file_fails_and_is_kept(tmp_path):
    (tmp_path / "x.csv").write_bytes(b"stale\n")
    with pytest.raises(F.HashMismatch):
        F.fetch_source(_source(), tmp_path, downloader=_writer(PAYLOAD))
    assert (tmp_path / "x.csv").read_bytes() == b"stale\n"


def test_seed_reused_only_by_hash(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "renamed_copy.csv").write_bytes(PAYLOAD)
    (seed / "same_size_wrong.csv").write_bytes(b"col\n2\n")
    cache = tmp_path / "cache"
    res = F.fetch_source(_source(), cache, seed_dirs=[seed], offline=True)
    assert res.action == "seeded"
    assert res.detail.endswith("renamed_copy.csv")
    assert (cache / "x.csv").read_bytes() == PAYLOAD


def test_offline_without_cache_or_seed_raises(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "wrong.csv").write_bytes(b"col\n2\n")
    with pytest.raises(F.MissingInput):
        F.fetch_source(_source(), tmp_path / "cache", seed_dirs=[seed], offline=True)


def test_unpinned_sources_skipped_unless_allowed(tmp_path):
    src = parse_manifest(_manifest(_entry(status="not_fetched", sha256=None, bytes=None,
                                          retrieved_utc=None)))["x"]
    dl = _writer(PAYLOAD)
    assert F.fetch_source(src, tmp_path, downloader=dl).action == "skipped"
    assert dl.calls == []
    res = F.fetch_source(src, tmp_path, downloader=dl, allow_unpinned=True)
    assert res.action == "downloaded_unpinned"
    assert res.sha256 == hashlib.sha256(PAYLOAD).hexdigest()


def test_request_sources_always_skipped(tmp_path):
    src = parse_manifest(_manifest(_entry(status="request", url=None, filename=None, sha256=None,
                                          bytes=None, retrieved_utc=None)))["x"]
    assert F.fetch_source(src, tmp_path, allow_unpinned=True).action == "skipped"


def test_http_download_reads_file_urls(tmp_path):
    src_file = tmp_path / "remote.csv"
    src_file.write_bytes(PAYLOAD)
    cache = tmp_path / "cache"
    res = F.fetch_source(_source(url=src_file.as_uri()), cache)  # real urllib path
    assert res.action == "downloaded"
    assert F.sha256_file(cache / "x.csv") == hashlib.sha256(PAYLOAD).hexdigest()


def test_cli_exit_codes(tmp_path, capsys):
    remote = tmp_path / "remote.csv"
    remote.write_bytes(PAYLOAD)
    entry = _entry(sha256=hashlib.sha256(PAYLOAD).hexdigest(), bytes=len(PAYLOAD), url=remote.as_uri())
    mpath = tmp_path / "sources.yaml"
    mpath.write_text(yaml.safe_dump(_manifest(entry)))
    cache = tmp_path / "cache"
    assert F.main(["--manifest", str(mpath), "--cache-dir", str(cache), "--offline"]) == 1
    assert F.main(["--manifest", str(mpath), "--cache-dir", str(cache)]) == 0
    assert F.main(["--manifest", str(mpath), "--cache-dir", str(cache), "--offline"]) == 0
    bad = copy.deepcopy(entry)
    bad["sha256"] = "0" * 64
    mpath.write_text(yaml.safe_dump(_manifest(bad)))
    assert F.main(["--manifest", str(mpath), "--cache-dir", str(tmp_path / "c2")]) == 1
    assert "FAIL" in capsys.readouterr().err
