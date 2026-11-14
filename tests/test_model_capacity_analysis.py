from copy import deepcopy

from scripts.analyze_model_capacity_experiment import decide_conclusion


def comparison_fixture() -> dict:
    per_class = [
        {"class_name": name, "map50_95_delta": 0.01}
        for name in (
            "helmet",
            "vest",
            "no_helmet",
            "no_vest",
            "gloves",
            "no_gloves",
            "goggles",
            "no_goggle",
        )
    ]
    return {
        "overall": {"map50_95": {"delta": 0.03}},
        "size_recall": {"overall": {"tiny": {"delta": 0.06}, "small": {"delta": 0.02}}},
        "latency": {"inference": {"ratio": 2.0}},
        "per_class": per_class,
    }


def test_capacity_decision_selects_a_for_material_efficient_gain() -> None:
    assert decide_conclusion(comparison_fixture())["option"] == "A"


def test_capacity_decision_selects_b_for_modest_gain() -> None:
    comparison = deepcopy(comparison_fixture())
    comparison["overall"]["map50_95"]["delta"] = 0.01

    assert decide_conclusion(comparison)["option"] == "B"


def test_capacity_decision_selects_c_without_useful_gain() -> None:
    comparison = deepcopy(comparison_fixture())
    comparison["overall"]["map50_95"]["delta"] = -0.001
    comparison["size_recall"]["overall"]["tiny"]["delta"] = -0.01
    for row in comparison["per_class"]:
        row["map50_95_delta"] = -0.01

    assert decide_conclusion(comparison)["option"] == "C"
