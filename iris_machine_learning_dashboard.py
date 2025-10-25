import streamlit as st
import pandas as pd
import plotly.express as px
import time, platform
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# --- Page configuration ---
st.set_page_config(page_title="🌸 Iris ML Dashboard", layout="wide")

# Add vertical space before title
st.markdown("<br>", unsafe_allow_html=True)
st.title("🌸 Iris Machine Learning Dashboard")

# Description
st.markdown(
    "<p style='font-size:20px;'>Experiment, evaluate, and predict <b>Iris Species</b> with various Machine Learning algorithms interactively.</p>",
    unsafe_allow_html=True
)

# --- CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fceabb, #f8b500);
    background-attachment: fixed;
}
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    box-sizing: border-box;
}
h1, h2, h3, h4 {
    color: #4a2c2a;
    font-weight: 700;
}
button[kind="primary"] {
    background-color: #ff7e5f !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #eb6750 !important;
}
@media (prefers-color-scheme: dark) {
    .block-container {
        background-color: rgba(30, 30, 30, 0.85);
    }
    h1, h2, h3, h4 {
        color: #ffcc99;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Load data ---
url = "https://github.com/azizmuzaki4/iris-machine-learning-dashboard/raw/refs/heads/main/iris_dataset.xlsx"
df = pd.read_excel(url)
df.fillna(df.mean(numeric_only=True), inplace=True)
df['species'] = df['species'].fillna('Unknown')

# --- Round numeric columns ---
num_cols_df = df.select_dtypes(include='number').columns
df[num_cols_df] = df[num_cols_df].round(2)

# --- Color per species ---
species_colors = {
    'setosa': '#FFEBEE',
    'versicolor': '#E8F5E9',
    'virginica': '#E3F2FD',
    'Unknown': '#F3E5F5'
}
def color_species(row):
    return [f'background-color: {species_colors.get(row["species"], "#FFFFFF")}' for _ in row]

# === IRIS dataset table ===
styled_df = (
    df.style
      .apply(color_species, axis=1)
      .set_table_styles([
          {"selector": "thead th",
           "props": [("background-color", "#4A90E2"),
                     ("color", "white"),
                     ("font-weight", "bold"),
                     ("text-align", "center"),
                     ("border-bottom", "2px solid #2E6EB5")]}
      ])
      .set_properties(**{"text-align": "center", "border": "1px solid #ddd"})
      .format({col: "{:.2f}" for col in num_cols_df if col in df.columns})
)

# === Statistical summary ===
summary_df = df.describe(include='all').transpose().reset_index().rename(columns={'index': 'Feature'})
for col in summary_df.columns:
    summary_df[col] = summary_df[col].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and pd.notnull(x) else x
    )
styled_summary = (
    summary_df.style
      .set_table_styles([
          {"selector": "thead th",
           "props": [("background-color", "#4A90E2"),
                     ("color", "white"),
                     ("font-weight", "bold"),
                     ("text-align", "center"),
                     ("border-bottom", "2px solid #2E6EB5")]}
      ])
      .set_properties(**{"text-align": "center", "border": "1px solid #ddd"})
)

# === CSS Freeze Header ===
freeze_css = """
<style>
.freeze-header thead th {
    position: sticky;
    top: 0;
    background-color: #4A90E2 !important;
    color: white !important;
    z-index: 2;
}
</style>
"""
st.markdown(freeze_css, unsafe_allow_html=True)

# === Two-column layout ===
col1, col2 = st.columns(2)
with col1:
    st.subheader("📂 IRIS Dataset")
    st.markdown(
        f'<div style="height:300px; overflow-y:scroll" class="freeze-header">{styled_df.to_html()}</div>',
        unsafe_allow_html=True
    )
with col2:
    st.subheader("📊 Statistical Summary")
    st.markdown(
        f'<div style="height:300px; overflow-y:scroll" class="freeze-header">{styled_summary.to_html()}</div>',
        unsafe_allow_html=True
    )

st.markdown("<hr style='border:2px solid #4A90E2; margin-top:25px; margin-bottom:25px;'>", unsafe_allow_html=True)

# --- Sidebar: Select scoring metric ---
st.sidebar.header("⚙ Tuning Settings")
scoring_option = st.sidebar.selectbox(
    "Select Metric for Tuning:",
    options=["Accuracy", "F1 Weighted", "Precision Weighted", "Recall Weighted"]
)
scoring_metric = {
    "Accuracy": "accuracy",
    "F1 Weighted": "f1_weighted",
    "Precision Weighted": "precision_weighted",
    "Recall Weighted": "recall_weighted"
}[scoring_option]

# --- Encode target ---
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# --- Split data ---
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model list + parameter grid ---
model_params = {
    "SVM (RBF)": (
        Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]),
        {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}
    ),
    "Random Forest": (
        Pipeline([('clf', RandomForestClassifier(random_state=42))]),
        {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 3, 5]}
    ),
    "Gradient Boosting": (
        Pipeline([('clf', GradientBoostingClassifier(random_state=42))]),
        {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.05, 0.1]}
    ),
    "Logistic Regression": (
        Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500, random_state=42))]),
        {'clf__C': [0.1, 1, 10]}
    ),
    "KNN": (
        Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        {'clf__n_neighbors': [3, 5, 7]}
    ),
    "Decision Tree": (
        Pipeline([('clf', DecisionTreeClassifier(random_state=42))]),
        {'clf__max_depth': [None, 3, 5]}
    ),
    "Naive Bayes": (
        Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
        {}
    )
}

# --- Button to redo tuning ---
if st.sidebar.button("🔄 Redo Tuning"):
    for key in ["tuning_done", "results_df", "best_models", "best_model_name",
                "best_model", "best_cv_model", "test_df"]:
        st.session_state.pop(key, None)

# --- Run tuning only once ---
if "tuning_done" not in st.session_state:
    results, best_models = [], {}
    progress_bar = st.progress(0)
    status_text, eta_text = st.empty(), st.empty()
    total_models, start_time = len(model_params), time.time()
    N_JOBS = -1 if platform.system().lower() != "windows" else 1

    for idx, (name, (pipeline, params)) in enumerate(model_params.items(), start=1):
        status_text.text(f"Processing: {name} ({idx}/{total_models}) ...")
        grid = GridSearchCV(pipeline, params, cv=5, scoring=scoring_metric, n_jobs=N_JOBS)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        results.append({"Model": name, "Best Params": grid.best_params_, "CV Score": grid.best_score_})
        progress_bar.progress(idx / total_models)
        elapsed = time.time() - start_time
        eta_text.text(f"⏳ Estimated time remaining: {elapsed/idx*(total_models-idx):.1f} seconds")

    status_text.text("✅ Finished tuning all models!")
    eta_text.text("")

    results_df = pd.DataFrame(results).sort_values(by="CV Score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_model_name]
    best_cv_model = results_df.iloc[0]["Model"]

    # Evaluate test set
    test_results = []
    for name, model in best_models.items():
        y_pred_test = model.predict(X_test)
        test_results.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, y_pred_test),
            "Test Precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            "Test Recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            "Test F1": f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        })
    test_df = pd.DataFrame(test_results).sort_values(by="Test Accuracy", ascending=False)

    # Save to session_state
    st.session_state.results_df = results_df
    st.session_state.best_models = best_models
    st.session_state.best_model_name = best_model_name
    st.session_state.best_model = best_model
    st.session_state.best_cv_model = best_cv_model
    st.session_state.test_df = test_df
    st.session_state.tuning_done = True

else:
    results_df = st.session_state.results_df
    best_models = st.session_state.best_models
    best_model_name = st.session_state.best_model_name
    best_model = st.session_state.best_model
    best_cv_model = st.session_state.best_cv_model
    test_df = st.session_state.test_df

# ==== Table styling function ====
def style_table(df, bar_subset=None, fmt=None, vmin=None, vmax=None):
    styler = df.style
    if fmt:
        styler = styler.format(fmt)
    styler = (
        styler.set_table_styles(
            [
                {"selector": "thead th",
                 "props": [("background-color", "#4A90E2"),
                           ("color", "white"),
                           ("font-weight", "bold"),
                           ("text-align", "center"),
                           ("border-bottom", "2px solid #2E6EB5")]},
                {"selector": "tbody td",
                 "props": [("text-align", "center"),
                           ("padding", "8px"),
                           ("border-bottom", "1px solid #ddd")]},
                {"selector": "tbody tr:nth-child(even)",
                 "props": [("background-color", "#F4F8FC")]},
                {"selector": "tbody tr:nth-child(odd)",
                 "props": [("background-color", "#FFFFFF")]},
                {"selector": "tbody tr:hover",
                 "props": [("background-color", "#D9E9FF")]}
            ]
        )
        .set_properties(**{"border-collapse": "collapse", "font-size": "14px"})
    )
    if bar_subset:
        styler = styler.bar(subset=bar_subset, color="#FF9800", vmin=vmin, vmax=vmax)
    return styler

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio(
    "Select Page:",
    ["🔍 Visualization", "⚙️ Training", "📊 Evaluation", "🔮 Prediction"]
)

# --- Visualization ---
if page == "🔍 Visualization":
    st.subheader("🔍 Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Filter & Histogram**")
        selected_species = st.multiselect(
            "Select Species:",
            df['species'].unique(),
            default=df['species'].unique()
        )
        feature = st.selectbox("Select feature for histogram:", X.columns)
    with col2:
        st.markdown("**Scatter Plot Controls**")
        x_axis = st.selectbox("X-axis (Scatter):", X.columns, key="xaxis")
        y_axis = st.selectbox("Y-axis (Scatter):", X.columns, key="yaxis")

    filtered_df = df[df['species'].isin(selected_species)]
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.histogram(filtered_df, x=feature, color="species", barmode="overlay"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            px.scatter(
                filtered_df, x=x_axis, y=y_axis, color="species", size="petal_width",
                hover_data=X.columns.tolist()+['species']
            ),
            use_container_width=True
        )

    st.markdown("### 📦 Boxplot per Feature")
    cols_per_row = 2
    feature_list = list(X.columns)
    for i in range(0, len(feature_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col_name in enumerate(feature_list[i:i+cols_per_row]):
            with cols[j]:
                fig = px.box(filtered_df, x="species", y=col_name, color="species", points="all")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔗 Scatter Matrix (Pairplot)")
    fig_matrix = px.scatter_matrix(filtered_df, dimensions=X.columns, color="species")
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("### 🔥 Feature Correlation")
    corr = filtered_df[X.columns].corr()
    try:
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    except TypeError:
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Training ---
elif page == "⚙️ Training":
    st.subheader("⚙️ Cross-Validation Results")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = px.bar(
            results_df,
            x="CV Score",
            y="Model",
            orientation="h",
            title="CV Score Comparison",
            text=results_df["CV Score"].apply(lambda x: f"{x:.2%}"),
            range_x=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        styled_html = style_table(
            results_df,
            bar_subset=["CV Score"],
            fmt={"CV Score": "{:.2%}"},
            vmin=results_df["CV Score"].min(),
            vmax=results_df["CV Score"].max()
        ).to_html()
        st.markdown(styled_html, unsafe_allow_html=True)

    best_cv_model = results_df.iloc[0]["Model"]
    best_cv_score = results_df.iloc[0]["CV Score"]
    worst_cv_model = results_df.iloc[-1]["Model"]
    worst_cv_score = results_df.iloc[-1]["CV Score"]
    gap_cv = best_cv_score - worst_cv_score

    st.markdown(f"""
    ### 🔍 Cross-Validation Insight
    - **{best_cv_model}** achieved the highest CV score of **{best_cv_score:.2%}**.
    - The model with the lowest score is **{worst_cv_model}** with a score of **{worst_cv_score:.2%}**.
    - The performance gap between the best and worst models in CV is about **{gap_cv:.2%}**.
    - **Note:** Based on CV score, {best_cv_model} is selected as the BEST MODEL for predicting IRIS species.
    """)

    with st.expander("🔍 Best Parameters"):
        st.json(results_df.set_index("Model")["Best Params"].to_dict())

    st.markdown("### 🕸 Radar Chart CV Score Comparison")
    radar_df = results_df.copy()
    radar_df["CV Score %"] = radar_df["CV Score"] * 100
    fig_radar = px.line_polar(radar_df, r="CV Score %", theta="Model", line_close=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Evaluation ---
elif page == "📊 Evaluation":
    st.subheader(f"🏆 Best Model Performance on Test Set")
    y_pred = best_model.predict(X_test)
    final_scores = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{final_scores['Accuracy']:.2%}")
    c2.metric("Precision", f"{final_scores['Precision']:.2%}")
    c3.metric("Recall", f"{final_scores['Recall']:.2%}")
    c4.metric("F1-Score", f"{final_scores['F1-Score']:.2%}")

    best_acc = test_df.iloc[0]["Test Accuracy"]
    worst_acc = test_df.iloc[-1]["Test Accuracy"]
    gap = best_acc - worst_acc
    prediction_correct = (best_model_name == best_cv_model)

    st.markdown(f"""
    ### 🔍 Test Set Evaluation Insight
    - The results above show the performance of {best_model_name} on the test data.
    - The model **{best_model_name}** was chosen for prediction because the model selection was purely based on the score in cross validation (training data).
    """)

    st.markdown("### 📊 Comparison of All Models on Test Set")
    styled_html2 = style_table(
        test_df,
        bar_subset=[c for c in test_df.columns if c != "Model"],
        fmt={col: "{:.2%}" for col in test_df.columns if col != "Model"},
        vmin=test_df.drop(columns="Model").min().min(),
        vmax=test_df.drop(columns="Model").max().max()
    ).to_html()
    st.markdown(styled_html2, unsafe_allow_html=True)

    from sklearn.metrics import confusion_matrix
    st.markdown("### 📉 Confusion Matrix for All Models")
    cols_per_row = 3
    model_names = list(best_models.keys())
    for i in range(0, len(model_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, name in enumerate(model_names[i:i+cols_per_row]):
            with cols[j]:
                y_pred_m = best_models[name].predict(X_test)
                acc_m = accuracy_score(y_test, y_pred_m)
                cm = confusion_matrix(y_test, y_pred_m, labels=range(len(le.classes_)))
                cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
                st.markdown(f"**{name}**  \nAccuracy: **{acc_m:.2%}**")
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
                fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{name}")

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    st.markdown("### 📈 ROC Curve for All Models (Multi-class)")
    fig_roc = go.Figure()
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    n_classes = y_test_bin.shape[1]
    for name, model in best_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            from sklearn.preprocessing import MinMaxScaler
            y_score = MinMaxScaler().fit_transform(scores)
        else:
            continue
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name} - {le.classes_[i]} (AUC={roc_auc:.2f})"
            ))
    fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=600)
    st.plotly_chart(fig_roc, use_container_width=True)

# --- Prediction ---
elif page == "🔮 Prediction":
    st.subheader("🔮 Species Prediction")
    col1, col2, col3, col4 = st.columns(4)
    sl = col1.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.0)
    sw = col2.number_input("Sepal Width", min_value=2.0, max_value=4.5, value=3.5)
    pl = col3.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4)
    pw = col4.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2)

    if st.button("Predict Now 🚀"):
        input_data = [[sl, sw, pl, pw]]
        pred_class = best_model.predict(input_data)[0]
        pred_species = le.inverse_transform([pred_class])[0]
        pred_proba = best_model.predict_proba(input_data)[0]
        top_n = sorted(
            [(le.inverse_transform([i])[0], p) for i, p in enumerate(pred_proba)],
            key=lambda x: x[1], reverse=True
        )
        st.success(f"Predicted Species: **{pred_species}**")
        st.write("🔝 **Top 3 Predictions:**")
        for sp, prob in top_n[:3]:
            st.write(f"- {sp}: {prob:.2%}")

        col_gauge, col_scatter = st.columns(2)
        with col_gauge:
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(pred_proba)*100,
                title={'text': "Confidence (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_scatter:
            fig_scatter = px.scatter(
                df, x="sepal_length", y="petal_length", color="species",
                title="Input Position in Feature Space", opacity=0.6
            )
            fig_scatter.add_scatter(
                x=[sl], y=[pl], mode="markers",
                marker=dict(color="black", size=12, symbol="x"), name="Input"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        mean_vals = df[df['species'] == pred_species][['sepal_length','sepal_width','petal_length','petal_width']].mean()
        st.markdown(f"""
        <div style="background-color:rgba(255, 249, 230, 0.9); padding:15px; border-radius:10px; border-left:6px solid #ff9800;">
        <span style="display:inline-block; background-color:#27ae60; color:white; padding:4px 10px; border-radius:12px; font-size:0.85em; font-weight:bold; margin-bottom:8px;">
        ✅ FINAL STEP
        </span>
        <h3>📝 Prediction Narrative</h3>
        <p>Based on the entire process carried out in this dashboard:</p>
        <ol>
        <li><b>🏆 Best Model Selection</b><br>
        From various algorithms tested in <b>Tab 2 (Cross-Validation)</b> and <b>Tab 3 (Test Set Evaluation)</b>,  
        the model <b style="color:#d35400;">{best_model_name}</b> was selected as the model with the best performance.</li>
        <li><b>🤖 Model Usage for Prediction</b><br>
        This model is used to predict the <i>species</i> of iris flowers based on the feature inputs you entered above.</li>
        <li><b>📊 Prediction Results</b><br>
        <ul>
        <li>Main prediction: <b style="color:#27ae60;">{pred_species}</b> with confidence level <b>{max(pred_proba):.2%}</b>.</li>
        <li>Comparison of your input values with the average of class <b>{pred_species}</b>:
            <ul>
            <li>Sepal Length: {sl} (average {mean_vals['sepal_length']:.2f})</li>
            <li>Sepal Width: {sw} (average {mean_vals['sepal_width']:.2f})</li>
            <li>Petal Length: {pl} (average {mean_vals['petal_length']:.2f})</li>
            <li>Petal Width: {pw} (average {mean_vals['petal_width']:.2f})</li>
            </ul>
        </li>
        </ul>
        </li>
        <li><b>📌 Visualization & Context</b><br>
        <ul>
        <li><b>Gauge</b> on the left shows the model's confidence level for the main prediction.</li>
        <li><b>Scatter plot</b> on the right shows your input position in the feature space compared to the training data.</li>
        </ul>
        </li>
        </ol>
        <p>💡 Thus, Tab 4 represents the <b>final stage</b> of the machine learning pipeline:<br>
        <span style="color:#2980b9;">Data Exploration → Best Model Selection → Evaluation → Final Prediction</span>.</p>
        </div>
        """, unsafe_allow_html=True)
