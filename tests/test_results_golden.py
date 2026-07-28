"""Golden-number and consistency tests over the committed results JSONs.

These catch silent regressions in the generating scripts: if a definition
changes (weighting, winner threshold, rate solving), the committed results
and these assertions must move together in one commit.
"""

import json
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
    for check in lvt["validation"]["checks"]:
        assert check["max_abs_delta_gap"] <= lvt["validation"]["tolerance_gbp"]
        assert check["share_households_with_benefit_interaction"] == 0.0
        assert check["model_poverty_bhc"] == pytest.approx(
            check["arithmetic_poverty_bhc"], abs=0.01
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
