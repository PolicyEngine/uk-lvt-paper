"""Golden-number and consistency tests over the committed results JSONs.

These catch silent regressions in the generating scripts: if a definition
changes (weighting, winner threshold, rate solving), the committed results
and these assertions must move together in one commit.
"""

import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lvt():
    return json.loads((ROOT / "results" / "lvt_results.json").read_text())


@pytest.fixture(scope="module")
def deep():
    return json.loads((ROOT / "results" / "deep_dive.json").read_text())


@pytest.fixture(scope="module")
def ext():
    return json.loads((ROOT / "results" / "extensions.json").read_text())


def test_budget_neutral_rate_consistent(lvt, deep, ext):
    assert lvt["council_tax_replacement"]["required_lvt_rate_pct"] == 0.79
    assert deep["budget_neutral_rate_pct"] == pytest.approx(0.786, abs=0.001)
    central = next(r for r in ext["robustness"] if r["label"] == "Central")
    assert central["rate_pct"] == pytest.approx(0.786, abs=0.001)


def test_validation_passed(lvt, ext):
    validation = lvt["validation"]
    assert validation["tolerance_float32_ulps"] == 8
    assert validation["tolerance_gbp"] == 1.0
    assert [c["rate"] for c in validation["checks"]] == [0.005, 0.01, 0.02]
    for check in validation["checks"]:
        # The guard: every household within 8 float32 spacings of its largest
        # total, so no household shows a benefit interaction.
        assert (
            check["max_delta_gap_float32_ulps"] <= validation["tolerance_float32_ulps"]
        )
        assert check["share_households_with_benefit_interaction"] == 0.0
        # The paper's statements (results.tex): the full model and the
        # arithmetic agree to within £0.25 for every household, and the share
        # differing by more than £1 is zero.
        assert check["max_abs_delta_gap"] <= 0.25
        assert check["share_households_gap_over_threshold_gbp"] == 0.0
        assert check["model_poverty_bhc"] == pytest.approx(
            check["arithmetic_poverty_bhc"], abs=0.01
        )
        assert check["model_poverty_ahc"] == pytest.approx(
            check["arithmetic_poverty_ahc"], abs=0.01
        )
    assert ext["validation"]["share_households_with_benefit_interaction"] == 0.0


def test_poverty_gini_conventions(lvt):
    pg = lvt["poverty_gini"]
    # Person-level HBAI poverty and equivalised person-weighted Gini.
    assert pg["baseline_poverty_bhc"] == pytest.approx(9.43, abs=0.01)
    assert pg["baseline_poverty_ahc"] == pytest.approx(14.63, abs=0.01)
    assert pg["baseline_gini"] == pytest.approx(0.2972, abs=0.0005)
    assert pg["baseline_wealth_gini"] == pytest.approx(0.7038, abs=0.0005)
    central = pg["scenarios"]["0.79%"]
    assert central["poverty_ahc_change"] == pytest.approx(-0.69, abs=0.01)
    assert abs(central["gini_change"]) < 0.005


def test_winner_share_central(ext):
    central = next(r for r in ext["robustness"] if r["label"] == "Central")
    assert central["pct_winners"] == pytest.approx(67.4, abs=0.1)
    # Revenue neutrality: aggregate change ~ 0.
    assert abs(central["aggregate_change_bn"]) < 0.05


def test_regional_impacts_reconcile_to_uk_total(lvt):
    regions = {row["region"]: row for row in lvt["impact_by_region"]}
    assert len(regions) == 12
    assert regions["Wales"]["avg_net_change"] == 608
    assert regions["Scotland"]["avg_net_change"] == 435
    assert regions["London"]["avg_net_change"] == -1125
    assert regions["East of England"]["avg_net_change"] == -96
    assert sum(row["aggregate_net_change_bn"] for row in regions.values()) == (
        pytest.approx(0.0, abs=0.05)
    )


def test_recycling_first_row_is_central_rate(deep):
    first = deep["recycling"]["scenarios"][0]
    assert first["rate_pct"] == pytest.approx(
        deep["budget_neutral_rate_pct"], abs=0.005
    )
    assert first["net_revenue_recycled_bn"] == 0.0
    assert first["dividend_per_person"] == 0


def test_progressivity_table(deep):
    prog = deep["progressivity"]
    ct, lvt_tax = prog["taxes"]
    assert ct["kakwani_wealth"] == pytest.approx(-0.523, abs=0.005)
    assert lvt_tax["kakwani_wealth"] == pytest.approx(-0.032, abs=0.005)
    # Bottom-coded sensitivity present and close to headline values.
    for t in (ct, lvt_tax):
        assert t["kakwani_wealth_bottom_coded"] == pytest.approx(
            t["kakwani_wealth"], abs=0.005
        )


def test_top_tail_reconciliation(deep):
    tt = deep["top_tail_reconciliation"]
    # Main-residence England count should be within ~20% of the OBR figure.
    assert (
        abs(
            tt["households_main_residence_over_2m_england_thousands"]
            - tt["obr_hvcts_properties_thousands"]
        )
        <= 0.2 * tt["obr_hvcts_properties_thousands"]
    )
    # All-residential-property counts exceed main-residence counts.
    assert (
        tt["households_residential_over_2m_thousands"]
        > tt["households_main_residence_over_2m_thousands"]
    )


def test_robustness_rows_present(ext):
    labels = {r["label"] for r in ext["robustness"]}
    assert {
        "Central",
        "OBR receipts target",
        "Great Britain only",
        "Household land only",
        "NI legacy land share",
        "OBR target on ONS-scaled base",
        "Proportional property tax",
    } <= labels
    variant_labels = {r["label"] for r in ext["incidence_variants"]}
    assert {"Corporate LVT as wealth charge", "50% rent pass-through"} <= variant_labels


def test_hvcts_comparator(deep):
    h = deep["hvcts"]
    assert h["properties_liable_thousands"] == pytest.approx(165, rel=0.2)
    assert h["share_borne_by_top_wealth_decile_pct"] == 100.0


# ---------------------------------------------------------------------------
# Headline numbers, pinned exactly to the committed JSON. If a regeneration
# moves any of these, the paper text has to move with it in the same commit.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extras():
    return json.loads((ROOT / "results" / "submission_extras.json").read_text())


def test_headline_numbers_pinned(lvt, deep, ext, extras):
    replacement = lvt["council_tax_replacement"]
    assert replacement["council_tax_revenue_bn"] == 58.5  # net council tax
    assert replacement["council_tax_gross_bn"] == 61.9
    assert replacement["council_tax_benefit_bn"] == 3.4
    assert replacement["total_land_bn"] == 7436.2
    assert replacement["required_lvt_rate_pct"] == 0.79
    assert replacement["households_m"] == 32.08
    # The budget-neutral rate to three decimals, wherever it is reported.
    assert deep["budget_neutral_rate_pct"] == 0.786
    assert ext["validation"]["central_rate_pct"] == 0.786
    assert extras["central_rate_pct"] == 0.786
    assert extras["replacement_target_bn"] == 58.5
    central = next(r for r in ext["robustness"] if r["label"] == "Central")
    assert central["rate_pct"] == 0.786
    assert central["revenue_target_bn"] == 58.5
    assert central["pct_winners"] == 67.4  # share of households gaining
    assert central["poverty_ahc_change"] == -0.69
    assert central["wealth_decile_10"] == -5871
    assert lvt["poverty_gini"]["scenarios"]["0.79%"]["poverty_ahc_change"] == -0.69


def test_headline_rate_is_net_council_tax_over_land(lvt):
    # 58.5 / 7436.2 = 0.787%, the rounded inputs; the unrounded ratio behind
    # the published 0.786 must round to the same two-decimal label.
    replacement = lvt["council_tax_replacement"]
    ratio_pct = (
        100 * replacement["council_tax_revenue_bn"] / replacement["total_land_bn"]
    )
    assert round(ratio_pct, 2) == replacement["required_lvt_rate_pct"]
    assert "0.79%" in lvt["poverty_gini"]["scenarios"]


# ---------------------------------------------------------------------------
# Provenance: every committed results JSON names the pinned model, core and
# dataset, and those pins agree with pyproject.toml and uv.lock.
# ---------------------------------------------------------------------------

PROVENANCE_STAMPED = (
    "lvt_results.json",
    "extensions.json",
    "deep_dive.json",
    "submission_extras.json",
    "bootstrap_ci.json",
    "lambda_mc.json",
)


def _simulation_pins() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = {}
    for spec in pyproject["project"]["optional-dependencies"]["simulation"]:
        name, version = spec.split("==")
        pins[name.strip()] = version.strip()
    return pins


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return {package["name"]: package["version"] for package in lock["package"]}


def test_simulation_pins_are_exact_and_locked():
    pins = _simulation_pins()
    assert pins["policyengine-uk"] == "2.89.4"
    assert set(pins) == {"policyengine-uk", "policyengine-core"}
    locked = _locked_versions()
    for name, version in pins.items():
        assert locked[name] == version


@pytest.mark.parametrize("name", PROVENANCE_STAMPED)
def test_results_carry_pinned_provenance(name):
    from uk_lvt.provenance import PINNED_DATASET, PINNED_DATASET_SHA256

    data = json.loads((ROOT / "results" / name).read_text())
    provenance = data["provenance"]
    locked = _locked_versions()
    for package, version in provenance["packages"].items():
        assert version == locked[package], package
    assert provenance["packages"]["policyengine-uk"] == "2.89.4"
    assert (
        provenance["packages"]["policyengine-core"]
        == _simulation_pins()["policyengine-core"]
    )
    assert provenance["dataset"] == PINNED_DATASET
    assert provenance["dataset_sha256"] == PINNED_DATASET_SHA256
    assert provenance["dataset_matches_pin"] is True


def test_lvt_results_top_level_stamp_matches_provenance(lvt):
    assert lvt["model_version"] == lvt["provenance"]["packages"]["policyengine-uk"]
    assert lvt["dataset"] == lvt["provenance"]["dataset"]
