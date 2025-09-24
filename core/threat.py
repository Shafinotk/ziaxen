def decide_threat(prob: float, label: str, low: float, high: float):
    benign = label.lower() == "normal"
    if benign:
        if prob >= high:
            return "normal", "Continue normal operation."
        elif prob >= low:
            return "normal", "Continue normal operation; keep monitoring."
        else:
            return "suspicious", "Limit speed; cross-check with redundant sensors."
    else:
        if prob >= high:
            return "attack", "Switch to safe mode: pull over, alert control center, restrict autonomous functions."
        elif prob >= low:
            return "suspicious", "Limit speed; increase following distance; reweight away from affected sensors."
        else:
            return "suspicious", "Limit speed; increase diagnostics; confirm with secondary checks."

def combine_decision(ego_res, lr_res):
    if ego_res is None and lr_res is None:
        return None
    rank = {"normal": 0, "suspicious": 1, "attack": 2}
    parts = [r for r in [ego_res, lr_res] if r is not None]
    worst = max(parts, key=lambda r: rank.get(r["threat_level"], 0))
    combined_level = worst["threat_level"]
    candidate = max(parts, key=lambda r: (0 if r["label"].lower()=="normal" else 1, r["prob"]))
    return {
        "label": candidate["label"],
        "prob": candidate["prob"],
        "threat_level": combined_level,
        "action": worst["action"],
        "rationale": "Combined by max severity across models."
    }
