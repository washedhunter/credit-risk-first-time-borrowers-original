from credit_risk.scoring.feature_info import format_value
from credit_risk.scoring.reasons import reason_codes


def test_groups_are_summed_and_ranked():
    contributions = {
        "EXT_SOURCE_1": 0.30,
        "EXT_SOURCE_2": 0.25,  # same group as EXT_SOURCE_1 -> one reason worth 0.55
        "CREDIT_INCOME_RATIO": 0.40,
        "YEARS_EMPLOYED": 0.10,
        "INST_LATE_RATE": -0.50,
        "FLAG_OWN_REALTY": -0.05,
        "BUR_COUNT": 0.01,
    }
    values = {
        "EXT_SOURCE_1": 0.12,
        "EXT_SOURCE_2": 0.2,
        "CREDIT_INCOME_RATIO": 6.5,
        "YEARS_EMPLOYED": 0.4,
        "INST_LATE_RATE": 0.0,
        "FLAG_OWN_REALTY": 1.0,
        "BUR_COUNT": 2.0,
    }
    risks, strengths = reason_codes(contributions, values, top_n=3)
    assert [r.code for r in risks] == ["external_scores", "affordability", "employment"]
    assert risks[0].impact == 0.55 and risks[0].top_feature == "EXT_SOURCE_1"
    assert "Low external credit scores" in risks[0].text and "0.12" in risks[0].text
    assert "Loan amount / income: 6.50" in risks[1].text
    assert [s.code for s in strengths] == ["hc_repayment", "region_housing"]
    assert "lowered the risk" in strengths[0].text


def test_missing_history_becomes_the_reason_only_when_truly_absent():
    contributions = {
        "PREV_AMT_ANNUITY_MEAN": 0.2,
        "EXT_SOURCE_1": 0.1,
        "BUR_AMT_MAX_OVERDUE_MAX": 0.05,
    }
    # First-time borrower: no bureau rows, no previous applications, no external scores.
    thin = {"BUR_COUNT": 0, "PREV_COUNT": 0}
    risks, _ = reason_codes(contributions, thin)
    texts = [r.text for r in risks]
    assert texts[0].startswith("No earlier Home Credit applications on file")
    assert texts[1].startswith("No external credit scores available")
    assert texts[2].startswith("No credit-bureau record")

    # Has history, but the top feature happens to be missing (never overdue, one score absent).
    has_history = {
        "BUR_COUNT": 2,
        "PREV_COUNT": 3,
        "EXT_SOURCE_2": 0.4,
        "PREV_AMT_ANNUITY_MEAN": None,
    }
    risks, _ = reason_codes(contributions, has_history)
    texts = " ".join(r.text for r in risks)
    assert "No credit-bureau record" not in texts
    assert "No earlier Home Credit applications" not in texts
    assert "No external credit scores" not in texts
    assert "not on file" in texts


def test_absence_as_a_strength():
    _, strengths = reason_codes({"CC_UTILISATION_MEAN": -0.3}, {"CC_MONTHS": None})
    assert strengths[0].text == "Credit-card usage: no record on file, which lowered the risk."


def test_format_value():
    assert format_value("AMT_CREDIT", 1234567.4) == "1,234,567"
    assert format_value("INST_LATE_RATE", 0.25) == "25%"
    assert format_value("BUR_DAYS_CREDIT_MAX", -30) == "30 days ago"
    assert format_value("YEARS_EMPLOYED", 2.0) == "2.0 years"
    assert format_value("FLAG_OWN_CAR", 1.0) == "yes"
    assert format_value("EXT_SOURCE_2", None) == "not available"
    assert format_value("NAME_INCOME_TYPE", "Working") == "Working"
