"""``hf://`` dataset sources resolve to the right repository, file and revision.

``uk_lvt.provenance`` hashes the dataset file a generator reads, so it must
request the same file from Hugging Face as the simulation does. These tests
need the pinned policyengine-core and policyengine-uk (the ``simulation``
extra), so the dev-only test run skips them and CI runs this file in a
separate step with that extra installed. No test downloads anything: the
Hugging Face download helpers are replaced with recorders.
"""

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

hf = pytest.importorskip("policyengine_core.tools.hugging_face")
uk_simulation = pytest.importorskip("policyengine_uk.simulation")

from uk_lvt.provenance import PINNED_DATASET, resolve_dataset_path  # noqa: E402

PINNED_REPO = "policyengine/policyengine-uk-data-private"
PINNED_FILE = "enhanced_frs_2023_24.h5"
PINNED_REVISION = "1.56.14"

# Timing on a busy shared host is not what these properties test.
NO_DEADLINE = settings(deadline=None)

segment = st.text(
    alphabet=string.ascii_lowercase + string.digits + "._-", min_size=1, max_size=12
)
revision = st.none() | segment


def _url(owner, repo, path, rev):
    return f"hf://{owner}/{repo}/{path}" + (f"@{rev}" if rev else "")


def _provenance_request(url):
    """(repo, file, revision) that ``resolve_dataset_path`` asks Hugging Face for."""
    calls = []

    def record(repo, repo_filename, version=None, local_dir=None):
        calls.append((repo, repo_filename, version))
        return "/not/downloaded"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hf, "download_huggingface_dataset", record)
        resolve_dataset_path(url)
    assert len(calls) == 1
    return calls[0]


def _simulation_request(url):
    """(repo, file, revision) that policyengine-uk's own download asks for."""
    calls = []

    def record(repo, repo_filename, version=None, local_dir=None):
        calls.append((repo, repo_filename, version))
        return "/not/downloaded"

    class NoBuild:
        def build_from_file(self, dataset_file, cache_key=None):
            pass

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(uk_simulation, "download_huggingface_dataset", record)
        uk_simulation.Simulation.build_from_url(NoBuild(), url)
    assert len(calls) == 1
    return calls[0]


def test_pin_resolves_to_its_repository_file_and_revision():
    assert _provenance_request(PINNED_DATASET) == (
        PINNED_REPO,
        PINNED_FILE,
        PINNED_REVISION,
    )


def test_pin_matches_the_simulations_own_request():
    assert _provenance_request(PINNED_DATASET) == _simulation_request(PINNED_DATASET)


def test_nested_path_keeps_owner_and_directories():
    url = f"hf://{PINNED_REPO}/subdir/{PINNED_FILE}@{PINNED_REVISION}"
    assert _provenance_request(url) == (
        PINNED_REPO,
        f"subdir/{PINNED_FILE}",
        PINNED_REVISION,
    )


@settings(NO_DEADLINE, max_examples=200)
@given(segment, segment, st.lists(segment, max_size=3), segment, revision)
def test_any_hf_url_keeps_owner_repository_path_and_revision(
    owner, repo, directories, filename, rev
):
    path = "/".join([*directories, filename])
    assert _provenance_request(_url(owner, repo, path, rev)) == (
        f"{owner}/{repo}",
        path,
        rev,
    )


@settings(NO_DEADLINE, max_examples=200)
@given(segment, segment, segment, revision)
def test_top_level_file_matches_the_simulations_own_request(owner, repo, filename, rev):
    # Differential check against policyengine-uk 2.89.4: for a file at the top
    # of the repository, as the pin is, the file hashed is the file the
    # simulation downloads. (For a file in a subdirectory policyengine-uk
    # requests the wrong repository; see ``resolve_dataset_path``.)
    url = _url(owner, repo, filename, rev)
    assert _provenance_request(url) == _simulation_request(url)
