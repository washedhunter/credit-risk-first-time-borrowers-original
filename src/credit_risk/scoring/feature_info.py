"""Plain-English label, reason group and unit for every model feature.

Reason codes sum SHAP values within a group, so three highly related features (say, the three
external scores) produce one reason, not three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# group key -> (short label, sentence used when the group pushes risk UP)
GROUPS: dict[str, tuple[str, str]] = {
    "external_scores": ("External credit scores", "Low external credit scores"),
    "affordability": (
        "Loan size and affordability",
        "Loan amount or repayments are high relative to income",
    ),
    "bureau_arrears": (
        "Arrears with other lenders",
        "Overdue payments or bad debt reported by the credit bureau",
    ),
    "bureau_history": (
        "Credit-bureau history",
        "Profile of loans with other lenders (number, age or outstanding debt)",
    ),
    "hc_repayment": (
        "Repayment record at Home Credit",
        "Late or short payments on earlier Home Credit loans",
    ),
    "hc_applications": (
        "Earlier Home Credit applications",
        "Pattern of earlier Home Credit applications (refusals, amounts or timing)",
    ),
    "card_usage": (
        "Credit-card usage",
        "High credit-card balances, drawings or minimum-only payments",
    ),
    "employment": (
        "Employment, income type and education",
        "Short employment history, or the income type, occupation or education profile",
    ),
    "stability": (
        "Recent changes and address checks",
        "Recent ID, registration or phone changes, or address mismatches",
    ),
    "region_housing": ("Region, housing and assets", "Region rating, housing or asset profile"),
    "loan_product": ("Loan type and paperwork", "Loan type, goods price or documents provided"),
    "social_circle": ("Social circle", "Defaults observed in the applicant's social circle"),
}


@dataclass(frozen=True)
class FeatureInfo:
    label: str
    group: str
    kind: str  # amount | ratio | share | days | years | count | score | flag | category | number


def _f(label: str, group: str, kind: str) -> FeatureInfo:
    return FeatureInfo(label, group, kind)


_BUILDING = {
    "APARTMENTS_AVG": "apartment size",
    "BASEMENTAREA_AVG": "basement area",
    "YEARS_BEGINEXPLUATATION_AVG": "years in use",
    "YEARS_BUILD_AVG": "building age",
    "COMMONAREA_AVG": "common area",
    "ELEVATORS_AVG": "elevators",
    "ENTRANCES_AVG": "entrances",
    "FLOORSMAX_AVG": "max floors",
    "FLOORSMIN_AVG": "min floors",
    "LANDAREA_AVG": "land area",
    "LIVINGAPARTMENTS_AVG": "living apartments",
    "LIVINGAREA_AVG": "living area",
    "NONLIVINGAPARTMENTS_AVG": "non-living apartments",
    "NONLIVINGAREA_AVG": "non-living area",
    "TOTALAREA_MODE": "total area",
}

FEATURES: dict[str, FeatureInfo] = {
    # --- application: categorical
    "NAME_CONTRACT_TYPE": _f("Contract type", "loan_product", "category"),
    "NAME_TYPE_SUITE": _f("Accompanied by", "loan_product", "category"),
    "NAME_INCOME_TYPE": _f("Income type", "employment", "category"),
    "NAME_EDUCATION_TYPE": _f("Education", "employment", "category"),
    "NAME_HOUSING_TYPE": _f("Housing", "region_housing", "category"),
    "OCCUPATION_TYPE": _f("Occupation", "employment", "category"),
    "ORGANIZATION_TYPE": _f("Employer type", "employment", "category"),
    # --- application: numeric
    "FLAG_OWN_CAR": _f("Owns a car", "region_housing", "flag"),
    "FLAG_OWN_REALTY": _f("Owns property", "region_housing", "flag"),
    "AMT_INCOME_TOTAL": _f("Annual income", "affordability", "amount"),
    "AMT_CREDIT": _f("Loan amount", "affordability", "amount"),
    "AMT_ANNUITY": _f("Loan annuity (regular repayment)", "affordability", "amount"),
    "AMT_GOODS_PRICE": _f("Price of goods financed", "loan_product", "amount"),
    "REGION_POPULATION_RELATIVE": _f("Region population (normalised)", "region_housing", "number"),
    "YEARS_EMPLOYED": _f("Years in current job", "employment", "years"),
    "YEARS_REGISTRATION": _f("Years since registration changed", "stability", "years"),
    "YEARS_ID_PUBLISH": _f("Years since ID document changed", "stability", "years"),
    "YEARS_SINCE_PHONE_CHANGE": _f("Years since phone changed", "stability", "years"),
    "OWN_CAR_AGE": _f("Age of own car (years)", "region_housing", "number"),
    "FLAG_EMP_PHONE": _f("Gave an employer phone", "employment", "flag"),
    "FLAG_WORK_PHONE": _f("Gave a work phone", "stability", "flag"),
    "FLAG_PHONE": _f("Gave a home phone", "stability", "flag"),
    "FLAG_EMAIL": _f("Gave an email", "stability", "flag"),
    "REGION_RATING_CLIENT": _f("Region rating (1 best - 3 worst)", "region_housing", "number"),
    "REGION_RATING_CLIENT_W_CITY": _f("Region rating incl. city (1-3)", "region_housing", "number"),
    "REG_REGION_NOT_LIVE_REGION": _f(
        "Registered region differs from home region", "stability", "flag"
    ),
    "REG_REGION_NOT_WORK_REGION": _f(
        "Registered region differs from work region", "stability", "flag"
    ),
    "LIVE_REGION_NOT_WORK_REGION": _f("Home region differs from work region", "stability", "flag"),
    "REG_CITY_NOT_LIVE_CITY": _f("Registered city differs from home city", "stability", "flag"),
    "REG_CITY_NOT_WORK_CITY": _f("Registered city differs from work city", "stability", "flag"),
    "LIVE_CITY_NOT_WORK_CITY": _f("Home city differs from work city", "stability", "flag"),
    "EXT_SOURCE_1": _f("External score 1", "external_scores", "score"),
    "EXT_SOURCE_2": _f("External score 2", "external_scores", "score"),
    "EXT_SOURCE_3": _f("External score 3", "external_scores", "score"),
    **{
        k: _f(f"Building: {v} (normalised)", "region_housing", "number")
        for k, v in _BUILDING.items()
    },
    "OBS_30_CNT_SOCIAL_CIRCLE": _f(
        "Social circle: contacts observed (30 DPD)", "social_circle", "count"
    ),
    "DEF_30_CNT_SOCIAL_CIRCLE": _f("Social circle: defaults (30 DPD)", "social_circle", "count"),
    "OBS_60_CNT_SOCIAL_CIRCLE": _f(
        "Social circle: contacts observed (60 DPD)", "social_circle", "count"
    ),
    "DEF_60_CNT_SOCIAL_CIRCLE": _f("Social circle: defaults (60 DPD)", "social_circle", "count"),
    "FLAG_DOCUMENT_3": _f("Provided document 3", "loan_product", "flag"),
    "DOCUMENT_COUNT": _f("Number of documents provided", "loan_product", "count"),
    "AMT_REQ_CREDIT_BUREAU_MON": _f(
        "Bureau enquiries in the last month", "bureau_history", "count"
    ),
    "AMT_REQ_CREDIT_BUREAU_QRT": _f(
        "Bureau enquiries in the last quarter", "bureau_history", "count"
    ),
    "AMT_REQ_CREDIT_BUREAU_YEAR": _f(
        "Bureau enquiries in the last year", "bureau_history", "count"
    ),
    # --- credit bureau (other lenders)
    "BUR_COUNT": _f("Loans with other lenders", "bureau_history", "count"),
    "BUR_ACTIVE_COUNT": _f("Active loans with other lenders", "bureau_history", "count"),
    "BUR_CLOSED_COUNT": _f("Closed loans with other lenders", "bureau_history", "count"),
    "BUR_BAD_DEBT_COUNT": _f("Bureau loans sold or written off", "bureau_arrears", "count"),
    "BUR_RECENT_1Y_COUNT": _f("Bureau loans opened in the last year", "bureau_history", "count"),
    "BUR_CARD_COUNT": _f("Credit cards with other lenders", "bureau_history", "count"),
    "BUR_MICROLOAN_COUNT": _f("Microloans with other lenders", "bureau_history", "count"),
    "BUR_MORTGAGE_CAR_COUNT": _f(
        "Mortgages / car loans with other lenders", "bureau_history", "count"
    ),
    "BUR_OVERDUE_NOW_COUNT": _f("Bureau loans overdue now", "bureau_arrears", "count"),
    "BUR_DAYS_CREDIT_MAX": _f("Most recent bureau loan opened", "bureau_history", "days"),
    "BUR_DAYS_CREDIT_MIN": _f("Oldest bureau loan opened", "bureau_history", "days"),
    "BUR_DAYS_CREDIT_MEAN": _f("Average bureau loan opened", "bureau_history", "days"),
    "BUR_DAYS_ENDDATE_MAX": _f("Latest bureau loan end date", "bureau_history", "days"),
    "BUR_DAYS_UPDATE_MAX": _f("Last bureau update", "bureau_history", "days"),
    "BUR_DAY_OVERDUE_MAX": _f("Max days overdue with other lenders", "bureau_arrears", "count"),
    "BUR_AMT_MAX_OVERDUE_MAX": _f(
        "Largest amount ever overdue (bureau)", "bureau_arrears", "amount"
    ),
    "BUR_AMT_OVERDUE_SUM": _f("Amount overdue now (bureau)", "bureau_arrears", "amount"),
    "BUR_PROLONG_SUM": _f("Bureau loan extensions", "bureau_arrears", "count"),
    "BUR_AMT_CREDIT_SUM": _f("Total credit with other lenders", "bureau_history", "amount"),
    "BUR_AMT_CREDIT_MEAN": _f("Average bureau loan size", "bureau_history", "amount"),
    "BUR_ACTIVE_DEBT_SUM": _f("Outstanding debt with other lenders", "bureau_history", "amount"),
    "BUR_ACTIVE_CREDIT_SUM": _f("Active credit with other lenders", "bureau_history", "amount"),
    "BUR_ACTIVE_ANNUITY_SUM": _f("Repayments due to other lenders", "affordability", "amount"),
    "BUR_ACTIVE_RATE": _f("Share of bureau loans still active", "bureau_history", "share"),
    "BUR_ACTIVE_DEBT_RATIO": _f(
        "Outstanding / original on active bureau loans", "bureau_history", "share"
    ),
    "BUR_CARD_UTILISATION": _f(
        "Utilisation of cards with other lenders", "bureau_history", "share"
    ),
    "BUR_DAYS_LAST_CLOSED": _f("Last bureau loan closed", "bureau_history", "days"),
    "BUR_BB_LATE_12M_COUNT": _f(
        "Late months with other lenders (last 12)", "bureau_arrears", "count"
    ),
    "BUR_BB_WORST_STATUS": _f("Worst bureau delinquency bucket (0-5)", "bureau_arrears", "number"),
    "BUR_BB_MONTHS": _f("Months of bureau payment history", "bureau_history", "count"),
    "BUR_BB_LATE_RATE": _f("Share of bureau months paid late", "bureau_arrears", "share"),
    # --- previous Home Credit applications
    "PREV_COUNT": _f("Earlier Home Credit applications", "hc_applications", "count"),
    "PREV_APPROVED_COUNT": _f("Earlier applications approved", "hc_applications", "count"),
    "PREV_REFUSED_COUNT": _f("Earlier applications refused", "hc_applications", "count"),
    "PREV_CANCELED_COUNT": _f("Earlier applications cancelled", "hc_applications", "count"),
    "PREV_RECENT_1Y_COUNT": _f("Applications in the last year", "hc_applications", "count"),
    "PREV_ACTIVE_COUNT": _f("Home Credit loans still running", "hc_applications", "count"),
    "PREV_CASH_COUNT": _f("Earlier cash-loan applications", "hc_applications", "count"),
    "PREV_REVOLVING_COUNT": _f("Earlier revolving-loan applications", "hc_applications", "count"),
    "PREV_DAYS_DECISION_MAX": _f("Most recent earlier application", "hc_applications", "days"),
    "PREV_DAYS_DECISION_MIN": _f("First earlier application", "hc_applications", "days"),
    "PREV_DAYS_LAST_REFUSED": _f("Most recent refusal", "hc_applications", "days"),
    "PREV_AMT_APPLICATION_MEAN": _f(
        "Average amount applied for before", "hc_applications", "amount"
    ),
    "PREV_AMT_CREDIT_MEAN": _f("Average earlier approved loan", "hc_applications", "amount"),
    "PREV_AMT_CREDIT_MAX": _f("Largest earlier approved loan", "hc_applications", "amount"),
    "PREV_AMT_ANNUITY_MEAN": _f("Average earlier annuity", "hc_applications", "amount"),
    "PREV_CNT_PAYMENT_MEAN": _f(
        "Average earlier loan term (instalments)", "hc_applications", "number"
    ),
    "PREV_APP_CREDIT_RATIO_MEAN": _f(
        "Applied / granted amount (earlier)", "hc_applications", "ratio"
    ),
    "PREV_RATE_DOWN_PAYMENT_MEAN": _f(
        "Average earlier down-payment rate", "hc_applications", "share"
    ),
    "PREV_INTEREST_PROXY_MEAN": _f(
        "Earlier loans' interest (per instalment)", "hc_applications", "share"
    ),
    "PREV_DAYS_LAST_DUE_MAX": _f(
        "Last planned due date of earlier loans", "hc_applications", "days"
    ),
    "PREV_DAYS_TERMINATION_MAX": _f("Most recent earlier loan ended", "hc_applications", "days"),
    "PREV_APPROVED_RATE": _f("Share of earlier applications approved", "hc_applications", "share"),
    "PREV_REFUSED_RATE": _f("Share of earlier applications refused", "hc_applications", "share"),
    "PREV_HIGH_YIELD_RATE": _f(
        "Share of earlier loans at high interest", "hc_applications", "share"
    ),
    "PREV_LAST_AMT_ANNUITY": _f(
        "Annuity of the last earlier application", "hc_applications", "amount"
    ),
    "PREV_LAST_REFUSED": _f("Last earlier application was refused", "hc_applications", "flag"),
    # --- instalments on earlier Home Credit loans
    "INST_COUNT": _f("Instalments due so far", "hc_repayment", "count"),
    "INST_LOANS": _f("Loans with instalment history", "hc_repayment", "count"),
    "INST_MISSED_COUNT": _f("Instalments with no payment", "hc_repayment", "count"),
    "INST_DAYS_LATE_MEAN": _f("Average days late per instalment", "hc_repayment", "number"),
    "INST_DAYS_LATE_MAX": _f("Most days late on an instalment", "hc_repayment", "count"),
    "INST_DAYS_EARLY_MEAN": _f("Average days paid early", "hc_repayment", "number"),
    "INST_DAYS_LAST_DUE": _f("Last instalment due", "hc_repayment", "days"),
    "INST_LATE_RATE": _f("Share of instalments paid late", "hc_repayment", "share"),
    "INST_LATE_30_RATE": _f("Share of instalments 30+ days late", "hc_repayment", "share"),
    "INST_UNDERPAID_RATE": _f("Share of instalments underpaid", "hc_repayment", "share"),
    "INST_UNDERPAYMENT_RATIO": _f("Share of amount due left unpaid", "hc_repayment", "share"),
    "INST_LATE_RATE_1Y": _f("Share paid late in the last year", "hc_repayment", "share"),
    # --- POS / cash loan balances
    "POS_MONTHS": _f("Months of POS/cash loan history", "hc_repayment", "count"),
    "POS_LOANS": _f("POS/cash loans", "hc_repayment", "count"),
    "POS_DPD_MAX": _f("Max days past due (POS/cash)", "hc_repayment", "count"),
    "POS_DPD_DEF_MAX": _f("Max days past due beyond tolerance (POS/cash)", "hc_repayment", "count"),
    "POS_LATE_MONTH_RATE": _f("Share of POS/cash months in arrears", "hc_repayment", "share"),
    "POS_LATE_RATE_1Y": _f(
        "Share of POS/cash months in arrears (last year)", "hc_repayment", "share"
    ),
    "POS_REMAINING_INSTALMENTS": _f("Instalments still to pay (POS/cash)", "hc_repayment", "count"),
    "POS_ACTIVE_COUNT": _f("Active POS/cash loans", "hc_repayment", "count"),
    "POS_COMPLETED_COUNT": _f("Completed POS/cash loans", "hc_repayment", "count"),
    # --- Home Credit credit cards
    "CC_LOANS": _f("Home Credit credit cards", "card_usage", "count"),
    "CC_MONTHS": _f("Months of card history", "card_usage", "count"),
    "CC_UTILISATION_MEAN": _f("Average card utilisation", "card_usage", "share"),
    "CC_UTILISATION_MAX": _f("Peak card utilisation", "card_usage", "share"),
    "CC_UTILISATION_3M": _f("Card utilisation, last 3 months", "card_usage", "share"),
    "CC_DPD_MAX": _f("Max days past due on cards", "hc_repayment", "count"),
    "CC_BELOW_MIN_PAYMENT_RATE": _f(
        "Share of months paying below the minimum", "card_usage", "share"
    ),
    "CC_PAYMENT_TO_MIN_MEAN": _f("Card payment / minimum due", "card_usage", "ratio"),
    "CC_DRAWINGS_MEAN": _f("Average monthly card drawings", "card_usage", "amount"),
    "CC_ATM_DRAWINGS_MEAN": _f("Average monthly ATM withdrawals", "card_usage", "amount"),
    "CC_BALANCE_MEAN": _f("Average card balance", "card_usage", "amount"),
    "CC_LATE_MONTH_RATE": _f("Share of card months in arrears", "hc_repayment", "share"),
    "CC_BALANCE_LAST": _f("Current card balance", "card_usage", "amount"),
    "CC_LIMIT_LAST": _f("Current card limit", "card_usage", "amount"),
    # --- derived ratios (recomputed from the inputs above)
    "CREDIT_INCOME_RATIO": _f("Loan amount / income", "affordability", "ratio"),
    "ANNUITY_INCOME_RATIO": _f("Annuity / income", "affordability", "share"),
    "ANNUITY_CREDIT_RATIO": _f(
        "Annuity / loan amount (shorter term = higher)", "affordability", "share"
    ),
    "CREDIT_GOODS_RATIO": _f("Loan amount / goods price", "loan_product", "ratio"),
    "EXT_SOURCE_MEAN": _f("Average external score", "external_scores", "score"),
    "EXT_SOURCE_MIN": _f("Lowest external score", "external_scores", "score"),
    "EXT_SOURCE_COUNT": _f("External scores available", "external_scores", "count"),
    "BUREAU_DEBT_INCOME_RATIO": _f("Debt with other lenders / income", "affordability", "ratio"),
    "TOTAL_ANNUITY_INCOME_RATIO": _f("All repayments / income", "affordability", "share"),
    "ANNUITY_VS_LAST_PREV_RATIO": _f(
        "Annuity vs last earlier application", "hc_applications", "ratio"
    ),
}


def info(name: str) -> FeatureInfo:
    return FEATURES.get(name, FeatureInfo(name.replace("_", " ").title(), "loan_product", "number"))


def format_value(name: str, value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    kind = info(name).kind
    if kind == "category":
        return str(value)
    v = float(value)  # type: ignore[arg-type]
    if kind == "amount":
        return f"{v:,.0f}"
    if kind == "share":
        return f"{v:.0%}"
    if kind == "ratio":
        return f"{v:.2f}"
    if kind == "score":
        return f"{v:.2f}"
    if kind == "years":
        return f"{v:.1f} years"
    if kind == "days":
        days = round(v)
        if days <= 0:
            return f"{-days:,} days ago"
        return f"in {days:,} days"
    if kind == "count":
        return f"{v:,.0f}"
    if kind == "flag":
        return "yes" if v >= 0.5 else "no"
    return f"{v:.3g}"
