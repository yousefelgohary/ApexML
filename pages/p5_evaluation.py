"""
pages/p5_evaluation.py
ApexML-style evaluation page for safe pipelines and competition model selection.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.engine import feature_importance_table, binary_curve_data, evaluate_baseline, bundle_to_bytes


def _require_results():
    if st.session_state.get("model_results") is None or not st.session_state.get("model_bundles"):
        st.warning("⚠️ No safe-model results yet. Train models from **ML Studio** first.")
        st.stop()


def _best_standard_row(results: pd.DataFrame, task: str, metric: str):
    mapping = {
        "Accuracy": ("Test_Accuracy", False),
        "Balanced Accuracy": ("Test_Balanced_Accuracy", False),
        "Weighted F1": ("Test_F1_Weighted", False),
        "Macro F1": ("Test_F1_Macro", False),
        "RMSE": ("Test_RMSE", True),
        "MAE": ("Test_MAE", True),
        "R²": ("Test_R2", False),
    }
    col, asc = mapping.get(metric, ("Test_F1_Weighted" if task == "classification" else "Test_RMSE", task == "regression"))
    return results.sort_values(col, ascending=asc).iloc[0], col


def render():
    st.title("🏆 Model Evaluation & Leaderboard")
    st.markdown("Compare models, inspect stability and overfitting, and verify the predictor used for the final 10 cases.")
    _require_results()

    results = st.session_state["model_results"].copy()
    bundles = st.session_state["model_bundles"]
    details = st.session_state.get("model_details", {})
    split = st.session_state.get("split_data_safe")
    task = st.session_state.get("eval_task_type", "classification")
    metric = st.session_state.get("primary_metric") or ("Accuracy" if task == "classification" else "RMSE")
    competition = st.session_state.get("competition_run")
    final_bundle = st.session_state.get("final_bundle")
    best_name = st.session_state.get("best_model_name") or "Best Model"

    tab_lb, tab_diag, tab_curves, tab_feat, tab_nn, tab_export = st.tabs([
        "🏆 Leaderboard", "🔬 Diagnostics", "📈 ROC / Errors",
        "📊 Feature Importance", "🧠 Neural Network", "💾 Export"
    ])

    with tab_lb:
        st.subheader("🏆 Final Model")
        a, b, c, d = st.columns(4)
        a.metric("Winner", best_name)
        b.metric("Primary Metric", metric)
        c.metric("Task", task.title())
        if competition is not None:
            top = competition["leaderboard"].iloc[0]
            d.metric("Repeated CV", f"{float(top['CV_Mean']):.4f}")
        else:
            best_row, metric_col = _best_standard_row(results, task, metric)
            d.metric("Test Score", f"{float(best_row[metric_col]):.4f}")

        if competition is not None:
            st.success("Competition Boost ranks candidates with repeated Cross Validation, then refits the winner on all labeled rows for the external 10-case test.")
            st.subheader("🎯 Competition Ranking")
            lb = competition["leaderboard"].copy()
            show_cols = [c for c in ["Candidate", "Source", "Model_Family", "CV_Mean", "CV_Std", "Stability_Score", "Fit_Seconds"] if c in lb.columns]
            st.dataframe(lb[show_cols], use_container_width=True, hide_index=True)
            fig = px.bar(lb.head(12), x="Candidate", y="CV_Mean",
                         error_y="CV_Std" if "CV_Std" in lb else None,
                         color="Source" if "Source" in lb else None,
                         title=f"Repeated CV — {competition['metric_label']}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            exp = competition.get("expected_correct_out_of_10")
            if exp is not None:
                st.info(f"Repeated-CV estimate: **≈ {exp:.1f}/10 correct** if the hidden cases follow the same distribution. This is an estimate, not a guarantee.")

        st.markdown("---")
        st.subheader("📊 Holdout Model Comparison")
        st.dataframe(results, use_container_width=True, hide_index=True)

        if split is not None:
            X_train, X_test, y_train, y_test = split
            try:
                baseline = evaluate_baseline(X_train, X_test, y_train, y_test, task)
                with st.expander("No-Skill Baseline"):
                    st.dataframe(pd.DataFrame([baseline]), use_container_width=True, hide_index=True)
            except Exception:
                pass

        if task == "classification":
            cols = [c for c in ["Test_Accuracy", "Test_Balanced_Accuracy", "Test_F1_Weighted", "Test_F1_Macro", "Test_ROC_AUC"] if c in results.columns]
            if cols:
                melt = results.melt(id_vars="Model", value_vars=cols, var_name="Metric", value_name="Score")
                st.plotly_chart(px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], template="plotly_white"), use_container_width=True)
        elif "Test_RMSE" in results:
            st.plotly_chart(px.bar(results.sort_values("Test_RMSE"), x="Model", y="Test_RMSE", title="Test RMSE — Lower is Better", template="plotly_white"), use_container_width=True)

    with tab_diag:
        st.subheader("🔬 Generalization, Cross Validation & Cost")
        useful = [c for c in [
            "Model", "Generalization_Gap", "CV_F1_Weighted_Mean", "CV_F1_Weighted_Std",
            "CV_RMSE_Mean", "CV_RMSE_Std", "Fit_Seconds", "Predict_ms_per_1000", "Model_Size_KB"
        ] if c in results.columns]
        st.dataframe(results[useful], use_container_width=True, hide_index=True)

        if "Generalization_Gap" in results:
            fig = px.bar(results, x="Model", y="Generalization_Gap", title="Train–Test Generalization Gap", template="plotly_white")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("A large gap can indicate overfitting. Low train and test performance together can indicate underfitting.")

        failures = details.get("__failures__", []) if isinstance(details, dict) else []
        if failures:
            with st.expander(f"⚠️ Models that could not train ({len(failures)})"):
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

        if task == "classification":
            model_name = st.selectbox("Model for confusion matrix / report", list(bundles.keys()), key="diag_model")
            diag = details.get(model_name, {})
            if diag.get("confusion_matrix") is not None:
                enc = bundles[model_name].get("target_encoder")
                labels = [str(x) for x in enc.classes_] if enc is not None else None
                fig = px.imshow(diag["confusion_matrix"], text_auto=True, x=labels, y=labels,
                                labels={"x": "Predicted", "y": "Actual"}, color_continuous_scale="Teal",
                                title=f"Confusion Matrix — {model_name}")
                st.plotly_chart(fig, use_container_width=True)
            if diag.get("classification_report") is not None:
                st.dataframe(pd.DataFrame(diag["classification_report"]).T, use_container_width=True)
        else:
            model_name = st.selectbox("Model for regression errors", list(bundles.keys()), key="diag_reg_model")
            diag = details.get(model_name, {})
            if diag.get("y_test") is not None:
                y_true = np.asarray(diag["y_test"])
                pred = np.asarray(diag["test_pred"])
                c1, c2 = st.columns(2)
                with c1:
                    frame = pd.DataFrame({"Actual": y_true, "Predicted": pred})
                    fig = px.scatter(frame, x="Actual", y="Predicted", opacity=.5, title="Actual vs Predicted", template="plotly_white")
                    lo, hi = float(min(y_true.min(), pred.min())), float(max(y_true.max(), pred.max()))
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal"))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    frame = pd.DataFrame({"Predicted": pred, "Residual": y_true - pred})
                    fig = px.scatter(frame, x="Predicted", y="Residual", opacity=.5, title="Residuals", template="plotly_white")
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, use_container_width=True)

    with tab_curves:
        if task == "classification" and split is not None:
            _, X_test, _, y_test = split
            if len(np.unique(y_test)) == 2:
                st.subheader("📈 ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
                for name, bundle in bundles.items():
                    try:
                        curves = binary_curve_data(bundle, X_test, y_test)
                        roc = curves.get("roc")
                        if roc is not None:
                            fig_roc.add_trace(go.Scatter(x=roc["FPR"], y=roc["TPR"], mode="lines", name=name))
                    except Exception:
                        pass
                fig_roc.update_layout(template="plotly_white", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC comparison here is shown for binary classification.")
        elif task == "regression":
            st.info("Regression error diagnostics are available in the Diagnostics tab.")

    with tab_feat:
        st.subheader("📊 Feature Importance / Coefficients")
        name = st.selectbox("Select model", list(bundles.keys()), key="fi_safe")
        fi = feature_importance_table(bundles[name])
        if fi is None:
            st.info("This model does not expose direct feature importance or coefficients.")
        else:
            top = fi.head(25)
            fig = px.bar(top.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Teal", title=f"Top Features — {name}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.caption("Feature importance describes model association, not causation.")

    with tab_nn:
        if "Neural Network" not in bundles:
            st.info("Train Neural Network from ML Studio to inspect its architecture and loss curve.")
        else:
            diag = details.get("Neural Network", {})
            nn = diag.get("nn_architecture")
            if nn:
                st.subheader("🧠 Neural Network Architecture")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Input Neurons", nn.get("transformed_input_features", "—"))
                c2.metric("Hidden Layers", len(nn.get("hidden_layer_sizes", ())))
                c3.metric("Output Neurons", nn.get("output_neurons", "—"))
                c4.metric("Trainable Params", f"{nn.get('trainable_parameters', 0):,}")
                if nn.get("all_layers"):
                    st.code("Input → " + " → ".join(map(str, nn["all_layers"][1:-1])) + " → Output")
                st.caption(f"Architecture mode: {nn.get('mode')}")
            loss = diag.get("loss_curve")
            if loss is not None:
                frame = pd.DataFrame({"Iteration": np.arange(1, len(loss) + 1), "Loss": loss})
                st.plotly_chart(px.line(frame, x="Iteration", y="Loss", title="Training Loss", template="plotly_white"), use_container_width=True)
                st.metric("Iterations Used", diag.get("n_iter", "—"))

    with tab_export:
        st.subheader("💾 Export Final Model")
        if final_bundle is not None:
            st.success(f"🏆 Locked final model: **{best_name}**")
            st.download_button("📥 Download Final Pipeline Bundle", data=bundle_to_bytes(final_bundle),
                               file_name="ApexML_final_model_bundle.pkl", mime="application/octet-stream", use_container_width=True)
        st.download_button("📥 Download Leaderboard CSV", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="ApexML_leaderboard.csv", mime="text/csv", use_container_width=True)
        if competition is not None:
            st.download_button("📥 Download Competition CV Ranking", data=competition["leaderboard"].to_csv(index=False).encode("utf-8"),
                               file_name="ApexML_competition_cv_ranking.csv", mime="text/csv", use_container_width=True)
