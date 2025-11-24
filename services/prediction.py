import numpy as np
from core.preprocessing import ensure_2d, reshape_to_model, preprocess_lidar_radar, softmax_if_needed
from core.threat import decide_threat
from core.config import EGO_LABELS, LR_LABELS

def predict_model(bundle, X2d, labels, desc_map, low, high):
    if labels == LR_LABELS:
        X = preprocess_lidar_radar(X2d)
    else:
        X = ensure_2d(X2d)

    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)

    if labels == EGO_LABELS:
        X_in = X.reshape((1, X.shape[1], 1))
    elif labels == LR_LABELS:
        X_in = X.reshape((X.shape[0], X.shape[1], 1))
    else:
        X_in = reshape_to_model(X, bundle.input_shape)

    preds = bundle.model.predict(X_in)
    if isinstance(preds, list):
        preds = preds[0]
    if preds.ndim == 2 and preds.shape[0] > 1:
        preds = preds.mean(axis=0, keepdims=True)

    p = softmax_if_needed(preds[0])
    idx = int(np.argmax(p))
    label = labels[idx] if idx < len(labels) else str(idx)
    prob = float(p[idx])
    threat, action = decide_threat(prob, label, low, high)

    return {
        "label": label,
        "prob": prob,
        "threat_level": threat,
        "action": action,
        "probs": p.tolist(),
        "labels": labels,
        "description": desc_map.get(label, "")
    }
