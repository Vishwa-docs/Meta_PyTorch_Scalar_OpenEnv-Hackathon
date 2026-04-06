"""Synthetic data generator for the PolypharmacyEnv.

Generates:
  - data/lookups/drug_metadata.csv
  - data/lookups/ddi_rules.csv
  - data/lookups/beers_criteria.csv
  - data/processed/patients_polypharmacy.csv
"""

from __future__ import annotations

import csv
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOKUPS = ROOT / "data" / "lookups"
PROCESSED = ROOT / "data" / "processed"

# ── Drug catalogue ───────────────────────────────────────────────────────────

DRUGS = [
    # drug_id, generic_name, atc_class, high_risk, default, min, max
    ("DRUG_WARFARIN",        "warfarin",         "B01AA",  1, 5.0,  1.0, 10.0),
    ("DRUG_APIXABAN",        "apixaban",         "B01AF",  1, 5.0,  2.5, 10.0),
    ("DRUG_METFORMIN",       "metformin",        "A10BA",  0, 1000, 500, 2000),
    ("DRUG_GLIPIZIDE",       "glipizide",        "A10BB",  1, 5.0,  2.5, 20.0),
    ("DRUG_LISINOPRIL",      "lisinopril",       "C09AA",  0, 10.0, 2.5, 40.0),
    ("DRUG_AMLODIPINE",      "amlodipine",       "C08CA",  0, 5.0,  2.5, 10.0),
    ("DRUG_METOPROLOL",      "metoprolol",       "C07AB",  0, 50.0, 25.0,200.0),
    ("DRUG_DIGOXIN",         "digoxin",          "C01AA",  1, 0.25, 0.0625,0.5),
    ("DRUG_FUROSEMIDE",      "furosemide",       "C03CA",  0, 40.0, 20.0,160.0),
    ("DRUG_SPIRONOLACTONE",  "spironolactone",   "C03DA",  0, 25.0, 12.5, 50.0),
    ("DRUG_ATORVASTATIN",    "atorvastatin",     "C10AA",  0, 20.0, 10.0, 80.0),
    ("DRUG_SIMVASTATIN",     "simvastatin",      "C10AA",  0, 20.0, 10.0, 40.0),
    ("DRUG_OMEPRAZOLE",      "omeprazole",       "A02BC",  0, 20.0, 10.0, 40.0),
    ("DRUG_DIAZEPAM",        "diazepam",         "N05BA",  1, 5.0,  2.0, 10.0),
    ("DRUG_ALPRAZOLAM",      "alprazolam",       "N05BA",  1, 0.5,  0.25, 2.0),
    ("DRUG_AMITRIPTYLINE",   "amitriptyline",    "N06AA",  1, 25.0, 10.0, 75.0),
    ("DRUG_INSULIN_GLARGINE","insulin glargine", "A10AE",  1, 20.0, 10.0, 60.0),
    ("DRUG_PREDNISONE",      "prednisone",       "H02AB",  0, 10.0, 5.0,  60.0),
    ("DRUG_NAPROXEN",        "naproxen",         "M01AE",  1, 500,  250, 1000),
    ("DRUG_IBUPROFEN",       "ibuprofen",        "M01AE",  1, 400,  200,  800),
    ("DRUG_CLOPIDOGREL",     "clopidogrel",      "B01AC",  0, 75.0, 75.0, 75.0),
    ("DRUG_ASPIRIN",         "aspirin",          "B01AC",  0, 81.0, 81.0, 325.0),
    ("DRUG_HYDROCHLOROTHIAZIDE","HCTZ",          "C03AA",  0, 25.0, 12.5, 50.0),
    ("DRUG_DONEPEZIL",       "donepezil",        "N06DA",  0, 5.0,  5.0,  10.0),
    ("DRUG_GABAPENTIN",      "gabapentin",       "N03AX",  0, 300,  100, 1200),
    ("DRUG_TRAMADOL",        "tramadol",         "N02AX",  1, 50.0, 25.0, 200.0),
    ("DRUG_FLUOXETINE",      "fluoxetine",       "N06AB",  0, 20.0, 10.0, 60.0),
    ("DRUG_SERTRALINE",      "sertraline",       "N06AB",  0, 50.0, 25.0, 200.0),
    ("DRUG_CIPROFLOXACIN",   "ciprofloxacin",    "J01MA",  0, 500,  250, 750),
    ("DRUG_TAMSULOSIN",      "tamsulosin",       "G04CA",  0, 0.4,  0.4,  0.8),
    ("DRUG_CELECOXIB",       "celecoxib",        "M01AE",  0, 200,  100,  400),
    ("DRUG_NORTRIPTYLINE",   "nortriptyline",    "N06AA",  0, 25.0, 10.0, 75.0),
    ("DRUG_LOSARTAN",        "losartan",         "C09AA",  0, 50.0, 25.0, 100.0),
]

# ── DDI rules ────────────────────────────────────────────────────────────────

DDI_PAIRS: list[tuple[str, str, str, str, str, float]] = [
    # id1, id2, severity, mechanism, recommendation, base_risk_score
    ("DRUG_WARFARIN",   "DRUG_NAPROXEN",    "severe",   "Increased bleeding risk – NSAID inhibits platelet + anticoagulant",  "avoid_combination",  0.90),
    ("DRUG_WARFARIN",   "DRUG_IBUPROFEN",   "severe",   "Increased bleeding risk – NSAID + anticoagulant synergy",            "avoid_combination",  0.88),
    ("DRUG_WARFARIN",   "DRUG_ASPIRIN",     "moderate", "Additive antiplatelet + anticoagulant bleeding risk",                 "monitor_closely",    0.55),
    ("DRUG_WARFARIN",   "DRUG_FLUOXETINE",  "moderate", "SSRI increases serotonin and may potentiate bleeding",               "monitor_closely",    0.45),
    ("DRUG_WARFARIN",   "DRUG_CIPROFLOXACIN","moderate","CYP1A2 inhibition raises warfarin levels",                            "dose_adjust",        0.50),
    ("DRUG_APIXABAN",   "DRUG_NAPROXEN",    "severe",   "DOAC + NSAID – high bleeding risk",                                  "avoid_combination",  0.85),
    ("DRUG_APIXABAN",   "DRUG_ASPIRIN",     "moderate", "Additive bleeding risk with antiplatelet",                            "monitor_closely",    0.50),
    ("DRUG_DIGOXIN",    "DRUG_AMIODARONE",  "severe",   "Amiodarone increases digoxin levels – toxicity risk",                 "dose_adjust",        0.80),
    ("DRUG_DIGOXIN",    "DRUG_SPIRONOLACTONE","moderate","Spironolactone may raise digoxin levels",                             "monitor_closely",    0.40),
    ("DRUG_METFORMIN",  "DRUG_CIPROFLOXACIN","moderate","Fluoroquinolone may cause dysglycemia with metformin",                "monitor_closely",    0.35),
    ("DRUG_DIAZEPAM",   "DRUG_TRAMADOL",    "severe",   "CNS depression – benzodiazepine + opioid",                           "avoid_combination",  0.92),
    ("DRUG_ALPRAZOLAM",  "DRUG_TRAMADOL",   "severe",   "CNS depression – benzodiazepine + opioid",                           "avoid_combination",  0.91),
    ("DRUG_LISINOPRIL", "DRUG_SPIRONOLACTONE","moderate","Hyperkalemia risk – ACE-I + K-sparing diuretic",                     "monitor_closely",    0.48),
    ("DRUG_LISINOPRIL", "DRUG_NAPROXEN",    "moderate", "NSAID reduces ACE-I efficacy, renal risk",                            "monitor_closely",    0.42),
    ("DRUG_SIMVASTATIN","DRUG_AMLODIPINE",  "moderate", "CYP3A4 interaction increases statin exposure",                        "dose_adjust",        0.38),
    ("DRUG_ATORVASTATIN","DRUG_CIPROFLOXACIN","mild",   "Minor CYP interaction raising statin levels",                         "no_action",          0.15),
    ("DRUG_CLOPIDOGREL","DRUG_OMEPRAZOLE",  "moderate", "PPI reduces clopidogrel activation via CYP2C19",                     "dose_adjust",        0.45),
    ("DRUG_INSULIN_GLARGINE","DRUG_GLIPIZIDE","moderate","Additive hypoglycemia risk",                                         "monitor_closely",    0.50),
    ("DRUG_FLUOXETINE", "DRUG_TRAMADOL",    "severe",   "Serotonin syndrome risk – SSRI + serotonergic opioid",               "avoid_combination",  0.82),
    ("DRUG_AMITRIPTYLINE","DRUG_TRAMADOL",  "severe",   "Serotonin syndrome + CNS depression",                                "avoid_combination",  0.85),
    ("DRUG_METOPROLOL", "DRUG_DIGOXIN",     "moderate", "Additive bradycardia",                                               "monitor_closely",    0.40),
    ("DRUG_FUROSEMIDE", "DRUG_DIGOXIN",     "moderate", "Loop diuretic causes hypokalemia increasing digoxin toxicity risk",   "monitor_closely",    0.45),
    ("DRUG_PREDNISONE", "DRUG_NAPROXEN",    "moderate", "GI bleeding risk – corticosteroid + NSAID",                           "monitor_closely",    0.50),
    ("DRUG_PREDNISONE", "DRUG_WARFARIN",    "mild",     "Corticosteroid may alter INR",                                       "monitor_closely",    0.25),
]

# ── Beers criteria ───────────────────────────────────────────────────────────

BEERS_ENTRIES: list[tuple[str, str, str | None, str]] = [
    # drug_id, criterion_type, condition, rationale
    ("DRUG_DIAZEPAM",       "avoid",              None,       "Long-acting benzodiazepine: falls, fractures, cognitive impairment in elderly"),
    ("DRUG_ALPRAZOLAM",     "avoid",              None,       "Benzodiazepine: falls, fractures, cognitive impairment in elderly"),
    ("DRUG_AMITRIPTYLINE",  "avoid",              None,       "Strongly anticholinergic TCA: sedation, confusion, urinary retention in elderly"),
    ("DRUG_GLIPIZIDE",      "caution",            None,       "Sulfonylurea: hypoglycemia risk higher in elderly"),
    ("DRUG_NAPROXEN",       "avoid",              "CKD",      "NSAID contraindicated in CKD – renal deterioration, fluid retention"),
    ("DRUG_IBUPROFEN",      "avoid",              "CKD",      "NSAID contraindicated in CKD – renal deterioration, fluid retention"),
    ("DRUG_NAPROXEN",       "caution",            None,       "NSAID: GI bleeding and renal risk in elderly"),
    ("DRUG_IBUPROFEN",      "caution",            None,       "NSAID: GI bleeding and renal risk in elderly"),
    ("DRUG_DIGOXIN",        "dose_adjust",        None,       "Avoid doses > 0.125 mg/day in elderly – toxicity risk"),
    ("DRUG_TRAMADOL",       "avoid",              None,       "Opioid: CNS depression, falls, constipation in elderly"),
    ("DRUG_METFORMIN",      "dose_adjust",        "CKD",      "Reduce dose or avoid if eGFR < 30 – lactic acidosis risk"),
    ("DRUG_INSULIN_GLARGINE","caution",           None,       "Tight glycemic control increases hypoglycemia risk in elderly"),
    ("DRUG_PREDNISONE",     "avoid_in_condition", "DM",       "Corticosteroid worsens glycemic control in diabetes"),
    ("DRUG_DONEPEZIL",      "avoid_in_condition", "dementia", "Limited benefit, GI side effects; reassess regularly"),
    ("DRUG_CIPROFLOXACIN",  "caution",            None,       "Fluoroquinolone: tendon rupture, QT prolongation risk in elderly"),
]

# ── Conditions pool & constraints ────────────────────────────────────────────

ALL_CONDITIONS = ["HTN", "DM", "HF", "CKD", "AF", "COPD", "OA", "depression", "dementia", "GERD", "BPH", "neuropathy"]
EGFR_CATS = ["normal", "mild", "moderate", "severe"]
LIVER_CATS = ["normal", "impaired"]

# Drugs that make clinical sense per condition
CONDITION_DRUG_MAP: dict[str, list[str]] = {
    "HTN":        ["DRUG_LISINOPRIL", "DRUG_AMLODIPINE", "DRUG_METOPROLOL", "DRUG_HYDROCHLOROTHIAZIDE", "DRUG_FUROSEMIDE"],
    "DM":         ["DRUG_METFORMIN", "DRUG_GLIPIZIDE", "DRUG_INSULIN_GLARGINE"],
    "HF":         ["DRUG_FUROSEMIDE", "DRUG_SPIRONOLACTONE", "DRUG_METOPROLOL", "DRUG_LISINOPRIL", "DRUG_DIGOXIN"],
    "CKD":        ["DRUG_FUROSEMIDE", "DRUG_AMLODIPINE"],
    "AF":         ["DRUG_WARFARIN", "DRUG_APIXABAN", "DRUG_METOPROLOL", "DRUG_DIGOXIN"],
    "COPD":       ["DRUG_PREDNISONE"],
    "OA":         ["DRUG_NAPROXEN", "DRUG_IBUPROFEN", "DRUG_TRAMADOL", "DRUG_GABAPENTIN"],
    "depression": ["DRUG_FLUOXETINE", "DRUG_SERTRALINE", "DRUG_AMITRIPTYLINE"],
    "dementia":   ["DRUG_DONEPEZIL"],
    "GERD":       ["DRUG_OMEPRAZOLE"],
    "BPH":        ["DRUG_TAMSULOSIN"],
    "neuropathy": ["DRUG_GABAPENTIN", "DRUG_AMITRIPTYLINE"],
}


def _normalise_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _gen_drug_metadata(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug_id", "generic_name", "atc_class", "is_high_risk_elderly",
                     "default_dose_mg", "min_dose_mg", "max_dose_mg"])
        for row in DRUGS:
            w.writerow(row)


def _gen_ddi_rules(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug_id_1", "drug_id_2", "severity", "mechanism",
                     "recommendation", "base_risk_score"])
        for pair in DDI_PAIRS:
            a, b = _normalise_pair(pair[0], pair[1])
            w.writerow([a, b, pair[2], pair[3], pair[4], pair[5]])


def _gen_beers(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug_id", "criterion_type", "condition", "rationale"])
        for row in BEERS_ENTRIES:
            w.writerow([row[0], row[1], row[2] or "", row[3]])


def _gen_patients(out: Path, n_easy: int = 40, n_med: int = 40, n_hard: int = 40) -> None:
    """Generate synthetic patient episodes tagged by difficulty."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    drug_ids = [d[0] for d in DRUGS]

    # Build severity lookup for quick reference
    severe_pairs: set[tuple[str, str]] = set()
    for pair in DDI_PAIRS:
        if pair[2] == "severe":
            severe_pairs.add(_normalise_pair(pair[0], pair[1]))

    rows: list[list[str]] = []
    ep_counter = 0

    def _pick_conditions(n: int) -> list[str]:
        return rng.sample(ALL_CONDITIONS, min(n, len(ALL_CONDITIONS)))

    def _drugs_for_conditions(conds: list[str], target_n: int) -> list[str]:
        pool: list[str] = []
        for c in conds:
            pool.extend(CONDITION_DRUG_MAP.get(c, []))
        pool = list(dict.fromkeys(pool))  # deduplicate preserving order
        rng.shuffle(pool)
        selected = pool[:target_n]
        # Pad with random drugs if needed
        remaining = [d for d in drug_ids if d not in selected]
        while len(selected) < target_n and remaining:
            pick = rng.choice(remaining)
            remaining.remove(pick)
            selected.append(pick)
        return selected

    def _count_severe(meds: list[str]) -> int:
        count = 0
        for a, b in combinations(meds, 2):
            if _normalise_pair(a, b) in severe_pairs:
                count += 1
        return count

    def _baseline_risk(meds: list[str]) -> float:
        risk = 0.0
        for pair in DDI_PAIRS:
            a, b = _normalise_pair(pair[0], pair[1])
            if a in meds and b in meds:
                risk += pair[5]
        return min(risk / max(len(meds), 1), 1.0)

    # Easy episodes: 3-5 drugs, exactly 1 severe DDI
    for _ in range(n_easy):
        ep_counter += 1
        n_drugs = rng.randint(3, 5)
        conds = _pick_conditions(rng.randint(1, 3))
        # Ensure at least one severe DDI pair is present
        for attempt in range(50):
            meds = _drugs_for_conditions(conds, n_drugs)
            if _count_severe(meds) >= 1:
                break
        else:
            # Force a known severe pair
            sp = rng.choice(list(severe_pairs))
            meds = list(set(meds[:n_drugs - 2]) | {sp[0], sp[1]})[:n_drugs]

        age = rng.randint(65, 90)
        sex = rng.choice(["M", "F"])
        egfr = rng.choices(EGFR_CATS, weights=[4, 3, 2, 1])[0]
        liver = rng.choices(LIVER_CATS, weights=[8, 2])[0]
        br = round(_baseline_risk(meds), 4)
        rows.append([
            f"EP_{ep_counter:04d}", str(age), sex, ";".join(conds),
            egfr, liver, ";".join(meds), str(br), "easy",
        ])

    # Medium episodes: 6-10 drugs, multiple DDIs
    for _ in range(n_med):
        ep_counter += 1
        n_drugs = rng.randint(6, 10)
        conds = _pick_conditions(rng.randint(3, 5))
        meds = _drugs_for_conditions(conds, n_drugs)
        age = rng.randint(65, 92)
        sex = rng.choice(["M", "F"])
        egfr = rng.choices(EGFR_CATS, weights=[3, 3, 3, 1])[0]
        liver = rng.choices(LIVER_CATS, weights=[7, 3])[0]
        br = round(_baseline_risk(meds), 4)
        rows.append([
            f"EP_{ep_counter:04d}", str(age), sex, ";".join(conds),
            egfr, liver, ";".join(meds), str(br), "medium",
        ])

    # Hard episodes: 10-15 drugs, many issues, include critical drugs
    for _ in range(n_hard):
        ep_counter += 1
        n_drugs = rng.randint(10, 15)
        conds = _pick_conditions(rng.randint(4, 7))
        meds = _drugs_for_conditions(conds, n_drugs)
        # Ensure some critical drugs are present
        critical = ["DRUG_WARFARIN", "DRUG_INSULIN_GLARGINE", "DRUG_DIGOXIN"]
        for cd in rng.sample(critical, min(2, len(critical))):
            if cd not in meds and len(meds) < 15:
                meds.append(cd)
        age = rng.randint(70, 95)
        sex = rng.choice(["M", "F"])
        egfr = rng.choices(EGFR_CATS, weights=[2, 2, 3, 3])[0]
        liver = rng.choices(LIVER_CATS, weights=[6, 4])[0]
        br = round(_baseline_risk(meds), 4)
        rows.append([
            f"EP_{ep_counter:04d}", str(age), sex, ";".join(conds),
            egfr, liver, ";".join(meds), str(br), "hard",
        ])

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "age", "sex", "conditions", "eGFR_category",
                     "liver_function_category", "medication_ids",
                     "baseline_risk_score", "difficulty"])
        for r in rows:
            w.writerow(r)


def main() -> None:
    print("Generating drug_metadata.csv …")
    _gen_drug_metadata(LOOKUPS / "drug_metadata.csv")
    print("Generating ddi_rules.csv …")
    _gen_ddi_rules(LOOKUPS / "ddi_rules.csv")
    print("Generating beers_criteria.csv …")
    _gen_beers(LOOKUPS / "beers_criteria.csv")
    print("Generating patients_polypharmacy.csv …")
    _gen_patients(PROCESSED / "patients_polypharmacy.csv")
    print("Done.")


if __name__ == "__main__":
    main()
