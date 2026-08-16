"""
pages/p6_inference.py
ApexML inference page with a locked final model, single prediction, batch prediction,
and a dedicated exactly-10-cases competition workflow.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from core.engine import prepare_inference_features
from core.competition import predict_competition_bundle


def _require_model():
    if st.session_state.get("final_bundle") is None:
        st.warning("⚠️ No final model yet. Train models from **ML Studio** first.")
        st.stop()


def _prepare(raw: pd.DataFrame, bundle: dict, setup: dict):
    X = prepare_inference_features(
        raw,
        setup.get("excluded", []),
        setup.get("date_cols", []),
        setup.get("force_numeric", []),
        setup.get("force_cat", []),
    )
    needed = bundle["feature_columns"]
    missing = [c for c in needed if c not in X.columns]
    extras = [c for c in X.columns if c not in needed]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    return X[needed], extras


def _predict(bundle: dict, X: pd.DataFrame):
    if bundle.get("competition"):
        return predict_competition_bundle(bundle, X)
    pipe = bundle["pipeline"]
    pred = pipe.predict(X)
    probs = pipe.predict_proba(X) if bundle["task"] == "classification" and hasattr(pipe, "predict_proba") else None
    return pred, probs


def _decode(bundle: dict, pred):
    if bundle["task"] == "classification":
        enc = bundle.get("target_encoder")
        return enc.inverse_transform(np.asarray(pred).astype(int)) if enc is not None else pred
    return np.asarray(pred, dtype=float)


def _raw_feature_columns(df: pd.DataFrame, setup: dict):
    return [c for c in df.columns if c != setup["target"] and c not in setup.get("excluded", [])]


def _default_row(df: pd.DataFrame, columns):
    row = {}
    for col in columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            row[col] = float(num.median()) if num.notna().any() else 0.0
        else:
            mode = s.dropna().astype(str).mode()
            row[col] = str(mode.iloc[0]) if not mode.empty else ""
    return row


def _read_upload(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))


def render():
    st.title("🔮 Inference / Prediction")
    st.markdown("The final winner is selected in ML Studio/Evaluation. Use it here for live prediction or the doctor's 10 hidden cases.")
    _require_model()

    final_bundle = st.session_state["final_bundle"]
    bundles = st.session_state.get("model_bundles", {})
    setup = st.session_state.get("dataset_setup")
    df = st.session_state.get("df_original")
    if not setup or df is None:
        st.error("Dataset setup is missing. Return to ML Studio → Setup.")
        return

    bundle = final_bundle
    best_name = st.session_state.get("best_model_name", "Final Winner")

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.success(f"🏆 **Locked Final Model:** {best_name}")
    c2.metric("Target", bundle["target_name"])
    c3.metric("Task", bundle["task"].title())

    with st.expander("Advanced: use another benchmark model instead", expanded=False):
        if bundles:
            names = ["Final Winner"] + list(bundles.keys())
            chosen = st.selectbox("Inference model", names, key="advanced_inference_model")
            if chosen != "Final Winner":
                bundle = bundles[chosen]
                st.warning("You are overriding the locked final winner for this prediction session.")

    tab_single, tab_ten, tab_batch = st.tabs([
        "📝 Single Prediction", "🎯 Doctor's 10 Cases", "📂 Batch Prediction"
    ])

    raw_cols = _raw_feature_columns(df, setup)

    with tab_single:
        st.subheader("📝 Enter One New Case")
        defaults = _default_row(df, raw_cols)
        values = {}
        cols = st.columns(3)
        for i, feat in enumerate(raw_cols):
            with cols[i % 3]:
                s = df[feat]
                if pd.api.types.is_numeric_dtype(s):
                    num = pd.to_numeric(s, errors="coerce")
                    med = float(num.median()) if num.notna().any() else 0.0
                    values[feat] = st.number_input(feat, value=med, key=f"single_{feat}")
                else:
                    choices = s.dropna().astype(str).value_counts().head(100).index.tolist()
                    values[feat] = st.selectbox(feat, choices if choices else [""], key=f"single_{feat}")

        if st.button("🔮 Generate Prediction", type="primary", key="single_predict"):
            try:
                raw = pd.DataFrame([values])
                X, extras = _prepare(raw, bundle, setup)
                pred, probs = _predict(bundle, X)
                decoded = _decode(bundle, pred)
                st.markdown("---")
                if bundle["task"] == "classification":
                    st.markdown(
                        f'''<div style="background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(5,150,105,.04));border:3px solid #10B981;border-radius:16px;padding:24px;text-align:center;">
                        <div style="font-size:2.5rem;font-weight:800;color:#059669;">{decoded[0]}</div>
                        <div style="color:#6B7280;">Predicted Class</div></div>''',
                        unsafe_allow_html=True,
                    )
                    if probs is not None:
                        enc = bundle.get("target_encoder")
                        labels = [str(x) for x in enc.classes_] if enc is not None else [str(i) for i in range(probs.shape[1])]
                        pframe = pd.DataFrame({"Class": labels, "Probability": probs[0]})
                        st.plotly_chart(px.bar(pframe, x="Probability", y="Class", orientation="h", range_x=[0,1], color="Probability", color_continuous_scale="Teal", template="plotly_white"), use_container_width=True)
                else:
                    st.metric("Predicted Value", f"{float(decoded[0]):,.6g}")
            except Exception as exc:
                st.exception(exc)

    with tab_ten:
        st.subheader("🎯 Final Test — Exactly 10 Unseen Cases")
        st.markdown("Use the locked winner for the 10 cases from the doctor. Upload a file or paste/edit 10 rows directly.")
        source = st.radio("Input method", ["Upload CSV / Excel", "Paste / edit 10 rows"], horizontal=True, key="ten_source")
        raw_cases = None

        if source == "Upload CSV / Excel":
            uploaded = st.file_uploader("Upload the 10 cases", type=["csv", "xlsx", "xls"], key="ten_upload")
            if uploaded is not None:
                try:
                    raw_cases = _read_upload(uploaded)
                    st.dataframe(raw_cases, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.exception(exc)
        else:
            row = _default_row(df, raw_cols)
            initial = pd.DataFrame([row.copy() for _ in range(10)])
            config = {}
            for col in raw_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    choices = df[col].dropna().astype(str).value_counts().head(100).index.tolist()
                    if 0 < len(choices) <= 30:
                        config[col] = st.column_config.SelectboxColumn(col, options=choices)
            raw_cases = st.data_editor(initial, use_container_width=True, hide_index=False, num_rows="fixed", column_config=config, key="ten_editor")

        if raw_cases is not None:
            if len(raw_cases) != 10:
                st.error(f"This final-test mode requires exactly **10 rows**. Current rows: **{len(raw_cases)}**.")
            else:
                st.success("✅ 10/10 cases detected — ready.")
                if st.button("🎯 Predict All 10 Cases", type="primary", use_container_width=True, key="predict_ten"):
                    try:
                        X, extras = _prepare(raw_cases, final_bundle, setup)
                        pred, probs = _predict(final_bundle, X)
                        decoded = _decode(final_bundle, pred)
                        out = raw_cases.copy().reset_index(drop=True)
                        out.insert(0, "Case", np.arange(1, 11))
                        out[f"Prediction_{final_bundle['target_name']}"] = decoded
                        if probs is not None:
                            out["Confidence"] = np.max(probs, axis=1)
                            enc = final_bundle.get("target_encoder")
                            if enc is not None and probs.shape[1] <= 10:
                                for i, label in enumerate(enc.classes_):
                                    out[f"P_{label}"] = probs[:, i]
                        st.session_state["predictions_10"] = out
                        st.balloons()
                    except Exception as exc:
                        st.exception(exc)

        out = st.session_state.get("predictions_10")
        if out is not None:
            st.markdown("---")
            st.subheader("🏁 Final 10 Predictions")
            display = ["Case", f"Prediction_{final_bundle['target_name']}"] + (["Confidence"] if "Confidence" in out.columns else [])
            st.dataframe(out[display], use_container_width=True, hide_index=True,
                         column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%%")} if "Confidence" in out.columns else None)
            st.download_button("💾 Download 10 Predictions", out.to_csv(index=False).encode("utf-8"), file_name="ApexML_doctor_10_predictions.csv", mime="text/csv", use_container_width=True)

            with st.expander("After the doctor reveals the correct answers — calculate our score"):
                if final_bundle["task"] == "classification":
                    classes = [str(x) for x in final_bundle["target_encoder"].classes_]
                    actual = pd.DataFrame({"Case": np.arange(1,11), "Actual": [classes[0]] * 10})
                    actual = st.data_editor(actual, hide_index=True, num_rows="fixed", column_config={"Actual": st.column_config.SelectboxColumn("Actual", options=classes)}, key="ten_actual")
                    if st.button("Calculate Score", key="score_ten"):
                        pred_col = f"Prediction_{final_bundle['target_name']}"
                        correct = (out[pred_col].astype(str).reset_index(drop=True) == actual["Actual"].astype(str).reset_index(drop=True))
                        st.session_state["last_doctor_score"] = {"Correct": int(correct.sum()), "Accuracy": float(correct.mean())}
                    score = st.session_state.get("last_doctor_score")
                    if score:
                        a, b = st.columns(2)
                        a.metric("Correct", f"{score['Correct']}/10")
                        b.metric("Accuracy", f"{score['Accuracy']:.0%}")
                else:
                    actual = pd.DataFrame({"Case": np.arange(1,11), "Actual": [0.0] * 10})
                    actual = st.data_editor(actual, hide_index=True, num_rows="fixed", key="ten_actual_reg")
                    if st.button("Calculate Regression Score", key="score_ten_reg"):
                        pred_col = f"Prediction_{final_bundle['target_name']}"
                        y_true = pd.to_numeric(actual["Actual"], errors="coerce")
                        y_pred = pd.to_numeric(out[pred_col], errors="coerce")
                        st.session_state["last_doctor_score"] = {
                            "MAE": mean_absolute_error(y_true, y_pred),
                            "RMSE": mean_squared_error(y_true, y_pred) ** .5,
                            "R2": r2_score(y_true, y_pred),
                        }
                    score = st.session_state.get("last_doctor_score")
                    if score:
                        a, b, c = st.columns(3)
                        a.metric("MAE", f"{score['MAE']:.4g}")
                        b.metric("RMSE", f"{score['RMSE']:.4g}")
                        c.metric("R²", f"{score['R2']:.4f}")

    with tab_batch:
        st.subheader("📂 General Batch Prediction")
        uploaded = st.file_uploader("Upload CSV / Excel with the same raw feature columns", type=["csv", "xlsx", "xls"], key="batch_general")
        if uploaded is not None:
            try:
                raw = _read_upload(uploaded)
                X, extras = _prepare(raw, bundle, setup)
                pred, probs = _predict(bundle, X)
                decoded = _decode(bundle, pred)
                out = raw.copy().reset_index(drop=True)
                out[f"Prediction_{bundle['target_name']}"] = decoded
                if probs is not None:
                    out["Confidence"] = np.max(probs, axis=1)
                st.success(f"✅ Predictions generated for **{len(out):,} rows**.")
                st.dataframe(out.head(100), use_container_width=True, hide_index=True)
                st.download_button("💾 Download Predictions CSV", out.to_csv(index=False).encode("utf-8"), file_name="ApexML_predictions.csv", mime="text/csv", use_container_width=True)
            except Exception as exc:
                st.exception(exc)
