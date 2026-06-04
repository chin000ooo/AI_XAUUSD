"""
export.py — optional/advanced deployment bridge.

  * ONNX export of tree models (skl2onnx / onnxmltools) for native OnnxRun inside
    an MQL5 EA, with a Python-vs-ONNX parity check.
  * A sample JSON signal payload for a TradingView/MT5 webhook-queue bridge.

Everything here is best-effort: if the optional converters are missing it logs
a warning and returns a status dict — it never hard-crashes the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_to_onnx(gbm_classifier, n_features: int, path, backend: str | None = None) -> dict:
    """Export a fitted GBMClassifier's underlying estimator to ONNX.

    Returns a status dict {ok, path, error}. Tree-model conversion uses
    onnxmltools for LightGBM/XGBoost and skl2onnx for sklearn.
    """
    path = Path(path)
    est = getattr(gbm_classifier, "estimator_", gbm_classifier)
    backend = backend or getattr(gbm_classifier, "backend_", None)
    n_feat = int(getattr(est, "n_features_in_", n_features))  # trust the model's own count

    def _convert(opset):
        if backend == "lightgbm":
            # onnxmltools converters require THEIR own FloatTensorType class
            from onnxmltools.convert.common.data_types import FloatTensorType as MlFloat
            from onnxmltools.convert import convert_lightgbm
            return convert_lightgbm(est, initial_types=[("input", MlFloat([None, n_feat]))],
                                    target_opset=opset, zipmap=False)
        if backend == "xgboost":
            from onnxmltools.convert.common.data_types import FloatTensorType as MlFloat
            from onnxmltools.convert import convert_xgboost
            return convert_xgboost(est, initial_types=[("input", MlFloat([None, n_feat]))],
                                   target_opset=opset, zipmap=False)
        from skl2onnx.common.data_types import FloatTensorType as SkFloat
        from skl2onnx import convert_sklearn
        return convert_sklearn(est, initial_types=[("input", SkFloat([None, n_feat]))],
                               target_opset=opset, options={"zipmap": False})

    last_err = None
    for opset in (13, 12, 9):
        try:
            onx = _convert(opset)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"[OUTPUT SAVED] ONNX model -> {path} (opset {opset}, backend {backend})")
            return {"ok": True, "path": str(path), "error": None}
        except Exception as e:
            last_err = e
            continue
    print(f"[WARNING] ONNX export skipped ({backend}): {last_err}")
    return {"ok": False, "path": str(path), "error": str(last_err)}


def onnx_parity_check(gbm_classifier, onnx_path, X_sample, tolerance: float = 1e-2) -> dict:
    """Compare Python predict_proba vs onnxruntime on a sample.

    NOTE: LightGBM/XGBoost ONNX converters only accept FLOAT32 input, while the
    Python model infers in float64. On XAUUSD, price-scale features (EMA ~5000)
    exceed float32's ~7 significant digits, so a few tree splits can flip and
    probabilities diverge by ~1e-3..1e-2. That is expected, not a bug. For tight
    production parity, scale large-magnitude features before export. ``tolerance``
    defaults to 1e-2 (1% max probability difference)."""
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(4)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        Xf = np.asarray(X_sample, dtype=np.float32)
        out = sess.run(None, {"input": Xf})
        # find the probability output (2-D, n_classes columns)
        onnx_proba = None
        for arr in out:
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[0] == Xf.shape[0]:
                onnx_proba = a
                break
            if isinstance(arr, list):  # zipmap output
                try:
                    onnx_proba = np.array([[row[k] for k in sorted(row)] for row in arr])
                    break
                except Exception:
                    pass
        py_proba = gbm_classifier.predict_proba(X_sample)
        if onnx_proba is None:
            return {"ok": False, "max_abs_diff": float("nan"), "note": "could not parse ONNX proba output"}
        m = min(onnx_proba.shape[1], py_proba.shape[1])
        diff = float(np.max(np.abs(onnx_proba[:, :m] - py_proba[:, :m])))
        if diff < 1e-4:
            note = f"parity OK (max_abs_diff={diff:.2e})"
        elif diff < tolerance:
            note = (f"parity acceptable (max_abs_diff={diff:.2e}); small float32 rounding on "
                    f"large-magnitude price-scale features — scale features for tighter prod parity")
        else:
            note = f"parity DIVERGENCE (max_abs_diff={diff:.2e} >= {tolerance:.0e}) — investigate"
        return {"ok": diff < tolerance, "max_abs_diff": diff, "note": note}
    except Exception as e:
        return {"ok": False, "max_abs_diff": float("nan"), "note": f"parity check skipped: {e}"}


def make_signal_payload(symbol: str, side: str, confidence: float, meta_prob: float,
                        size_lots: float, sl: float, tp: float, ts) -> dict:
    """Build the JSON payload an EA poller would consume.

    NOTE: MT5 has no inbound webhook. The production pattern is a Flask/queue
    server that the EA polls (see README). Verify contract size / point value
    against the live broker before trusting ``size_lots`` / ``sl`` / ``tp``.
    """
    return {
        "symbol": symbol,
        "side": side,                       # "long" / "short" / "flat"
        "confidence": round(float(confidence), 4),
        "meta_prob": round(float(meta_prob), 4),
        "size_lots": round(float(size_lots), 2),
        "sl": round(float(sl), 3),
        "tp": round(float(tp), 3),
        "ts": str(ts),
        "schema": "xauusd-signal-v1",
    }


def save_signal_payload_sample(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OUTPUT SAVED] sample signal payload -> {path}")
    return path
