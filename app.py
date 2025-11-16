
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import base64

st.set_page_config(layout="wide", page_title="HR Attrition Dashboard")

st.title("HR Attrition Dashboard — Streamlit Cloud Ready")
st.markdown("Upload `vf.csv` (or use sample) and explore employee attrition insights. Tabs: Overview, Model Trainer, Predict.")

@st.cache_data(show_spinner=False)
def load_sample():
    # a tiny sample dataset with typical columns; user can upload real vf.csv
    df = pd.DataFrame({
        "EmployeeID":[1,2,3,4,5,6,7,8,9,10],
        "JobRole":["Sales","Technical","HR","Sales","Technical","HR","Sales","Technical","HR","Sales"],
        "Satisfaction":[0.2,0.8,0.6,0.4,0.9,0.3,0.5,0.7,0.4,0.6],
        "MonthlyIncome":[3000,7000,4500,3200,8000,4100,3800,7600,4200,3900],
        "YearsAtCompany":[1,5,3,2,8,4,1,6,3,2],
        "LastPromotionYears":[0,2,1,0,3,1,0,2,1,0],
        "Q38_Followup_Interest":["No","Yes","No","No","Yes","No","No","Yes","No","No"]
    })
    return df

def preprocess(df, label_col="Q38_Followup_Interest"):
    # Drop obvious id columns except EmployeeID kept optional
    X = df.copy()
    if label_col not in X.columns:
        raise ValueError(f"Label column {label_col} not found.")
    y = X[label_col].astype(str).copy()
    X = X.drop(columns=[label_col])
    # Identify types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    # Build preprocessor
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)], remainder='drop')
    # Fit-transform
    X_trans = preprocessor.fit_transform(X)
    # Get feature names
    feat_names = []
    feat_names.extend(numeric_cols)
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(categorical_cols))
        feat_names.extend(ohe_names)
    except Exception:
        # fallback
        feat_names.extend([f"cat_{i}" for i in range(X_trans.shape[1]-len(numeric_cols))])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X_trans, y_enc, preprocessor, le, feat_names, numeric_cols, categorical_cols

def train_models(X_train, y_train, cv=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    models = {
        "DecisionTree": (DecisionTreeClassifier(random_state=random_state), {"max_depth":[3,5,8,None], "min_samples_leaf":[1,3,5]}),
        "RandomForest": (RandomForestClassifier(random_state=random_state, n_jobs=-1), {"n_estimators":[100], "max_depth":[5,10,None], "min_samples_leaf":[1,3]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=random_state), {"n_estimators":[100], "learning_rate":[0.05,0.1], "max_depth":[3,5]})
    }
    fitted = {}
    for name,(est,params) in models.items():
        grid = GridSearchCV(estimator=est, param_grid=params, cv=skf, scoring='f1_weighted' if len(set(y_train))>2 else 'f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        fitted[name] = grid.best_estimator_
    return fitted

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_roc_all(models, X_test, y_test, le):
    plt.figure(figsize=(6,5))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.shape[1]==2:
                fpr, tpr, _ = roc_curve(y_test, proba[:,1])
                auc = roc_auc_score(y_test, proba[:,1])
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            else:
                # multiclass: micro-average
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
                fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
                auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
                plt.plot(fpr, tpr, label=f"{name} (macro-AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves (Test)")
    plt.legend(loc='lower right'); plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

def download_df(df, filename="predictions.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv, file_name=filename, mime='text/csv')

# Layout: sidebar - upload or use sample
st.sidebar.header("Data and Filters")
uploaded = st.sidebar.file_uploader("Upload vf.csv (or use sample)", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample dataset", value=True if uploaded is None else False)
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error("Error reading uploaded CSV. Using sample.")
        df = load_sample()
elif use_sample:
    df = load_sample()
else:
    st.sidebar.info("Upload a CSV or enable sample.")
    df = load_sample()

st.sidebar.markdown("### Filter controls (apply to Overview charts)")
# job role multiselect and satisfaction slider
if 'JobRole' in df.columns:
    roles = sorted(df['JobRole'].dropna().unique().tolist())
    selected_roles = st.sidebar.multiselect("Filter by JobRole (multi-select)", options=roles, default=roles)
else:
    selected_roles = None
satisfaction_col = None
for c in df.columns:
    if 'satisfaction' in c.lower():
        satisfaction_col = c
        break
if satisfaction_col:
    minv = float(df[satisfaction_col].min()); maxv = float(df[satisfaction_col].max())
    sat_range = st.sidebar.slider(f"Filter {satisfaction_col}", min_value=minv, max_value=maxv, value=(minv,maxv))
else:
    sat_range = None

# Apply filters to a working dataframe for overview charts
df_overview = df.copy()
if selected_roles is not None and 'JobRole' in df_overview.columns:
    df_overview = df_overview[df_overview['JobRole'].isin(selected_roles)]
if satisfaction_col and sat_range:
    df_overview = df_overview[df_overview[satisfaction_col].between(sat_range[0], sat_range[1])]

# Main tabs
tab1, tab2, tab3 = st.tabs(["Overview (Charts)", "Model Trainer", "Predict & Download"])

with tab1:
    st.header("Overview — 5 Actionable Charts")
    # Chart 1: Attrition rate by JobRole (bar)
    if 'Q38_Followup_Interest' in df_overview.columns and 'JobRole' in df_overview.columns:
        st.subheader("1) Attrition rate by Job Role")
        grp = df_overview.groupby('JobRole')['Q38_Followup_Interest'].apply(lambda x: (x=='Yes').mean()).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(x=grp.values, y=grp.index, ax=ax)
        ax.set_xlabel("Attrition Rate (proportion of 'Yes')"); ax.set_ylabel("Job Role")
        st.pyplot(fig); plt.close(fig)
        st.markdown("Action: focus retention for roles with high attrition rate.")

    # Chart 2: Satisfaction vs Attrition (boxplot) by JobRole
    if satisfaction_col and 'Q38_Followup_Interest' in df_overview.columns:
        st.subheader("2) Satisfaction distribution by Attrition status")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.boxplot(x='Q38_Followup_Interest', y=satisfaction_col, data=df_overview, ax=ax)
        ax.set_xlabel("Attrition"); ax.set_ylabel(satisfaction_col)
        st.pyplot(fig); plt.close(fig)
        st.markdown("Action: low satisfaction correlates with leavers — consider targeted engagement programs.")

    # Chart 3: Income distribution for leavers vs stayers (histograms)
    if 'MonthlyIncome' in df_overview.columns and 'Q38_Followup_Interest' in df_overview.columns:
        st.subheader("3) Income distribution — Leavers vs Stayers")
        fig, ax = plt.subplots(figsize=(6,3))
        for lbl in df_overview['Q38_Followup_Interest'].unique():
            sns.kdeplot(df_overview[df_overview['Q38_Followup_Interest']==lbl]['MonthlyIncome'].dropna(), label=str(lbl), fill=True, ax=ax)
        ax.set_xlabel("Monthly Income"); ax.legend(); st.pyplot(fig); plt.close(fig)
        st.markdown("Action: review compensation bands for roles with higher attrition.")

    # Chart 4: Years at company vs Attrition (scatter + trend)
    if 'YearsAtCompany' in df_overview.columns and 'Q38_Followup_Interest' in df_overview.columns:
        st.subheader("4) Years at Company vs Attrition probability")
        # compute probability by binned years
        df_overview['YearsBin'] = pd.cut(df_overview['YearsAtCompany'].fillna(0), bins=5)
        prob = df_overview.groupby('YearsBin')['Q38_Followup_Interest'].apply(lambda x: (x=='Yes').mean()).rename("attr_prob").reset_index()
        fig, ax = plt.subplots(figsize=(6,3))
        sns.pointplot(x=prob['YearsBin'].astype(str), y=prob['attr_prob'], ax=ax)
        ax.set_xlabel("Years at Company (binned)"); ax.set_ylabel("Attrition Probability"); plt.xticks(rotation=45)
        st.pyplot(fig); plt.close(fig)
        st.markdown("Action: identify critical tenure ranges for interventions.")

    # Chart 5: Top drivers (simple feature importance proxy using correlation with target for numeric features)
    st.subheader("5) Quick Drivers (numeric features correlated with attrition)")
    if 'Q38_Followup_Interest' in df_overview.columns:
        # encode target to numeric temporarily
        df_tmp = df_overview.copy()
        df_tmp['__target__'] = (df_tmp['Q38_Followup_Interest']=='Yes').astype(int)
        num_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols)>1:
            corrs = df_tmp[num_cols].corr()['__target__'].abs().sort_values(ascending=False).drop('__target__')
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=corrs.values, y=corrs.index, ax=ax)
            ax.set_xlabel("Absolute correlation with Attrition (Yes=1)"); st.pyplot(fig); plt.close(fig)
            st.markdown("Action: deep-dive the top correlated features for targeted policies.")
        else:
            st.info("Not enough numeric columns to compute driver correlations.")

with tab2:
    st.header("Model Trainer — Train all 3 algorithms (CV=5, stratified)")
    st.markdown("This tab runs Decision Tree, Random Forest, Gradient Boosting using stratified 5-fold CV. It uses label `Q38_Followup_Interest`.")
    if 'Q38_Followup_Interest' not in df.columns:
        st.error("Label column `Q38_Followup_Interest` not found in dataset. Upload a dataset with this label.")
    else:
        st.markdown("Select features to use (optional):")
        all_features = [c for c in df.columns if c!='Q38_Followup_Interest']
        sel = st.multiselect("Features (default = all)", options=all_features, default=all_features)
        run_button = st.button("Train models (may take time)")
        if run_button:
            with st.spinner("Preprocessing and training models..."):
                subdf = df[sel + ['Q38_Followup_Interest']].dropna(axis=0, how='all')
                X, y, preprocessor, le, feat_names, num_cols, cat_cols = preprocess(subdf)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                models = train_models(X_train, y_train, cv=5)
                st.success("Training complete. Showing metrics...")
                # Metrics table
                metrics = []
                for name, m in models.items():
                    y_pred = m.predict(X_test)
                    train_pred = m.predict(X_train)
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    try:
                        if hasattr(m, "predict_proba") and m.predict_proba(X_test).shape[1]==2:
                            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
                        else:
                            auc = np.nan
                    except Exception:
                        auc = np.nan
                    metrics.append({"Algorithm":name, "Train Acc":train_acc, "Test Acc":test_acc, "Precision":precision, "Recall":recall, "F1":f1, "AUC":auc})
                mdf = pd.DataFrame(metrics).set_index("Algorithm").round(4)
                st.dataframe(mdf)
                # Show ROC
                st.subheader("ROC Curves")
                plot_roc_all(models, X_test, y_test, le)
                # Confusion matrices and feature importances
                for name, m in models.items():
                    st.subheader(f"{name} — Confusion Matrices")
                    plot_confusion_matrix(y_train, m.predict(X_train), le.classes_, title=f"{name} - Train")
                    plot_confusion_matrix(y_test, m.predict(X_test), le.classes_, title=f"{name} - Test")
                    # feature importances if available
                    if hasattr(m, "feature_importances_"):
                        st.subheader(f"{name} — Top feature importances")
                        importances = pd.Series(m.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
                        st.bar_chart(importances)
                # Save models and preprocessor to session state for prediction tab
                st.session_state['models'] = models
                st.session_state['preprocessor'] = preprocessor
                st.session_state['label_encoder'] = le
                st.session_state['feature_names'] = feat_names
                st.session_state['selected_features'] = sel

with tab3:
    st.header("Predict on new data & download results")
    st.markdown("Upload a dataset (same columns used for training). If models not trained here, you can still upload an external pre-trained model in pickle format.")
    upload_new = st.file_uploader("Upload dataset for prediction (CSV)", type=['csv'])
    model_upload = st.file_uploader("Or upload a trained models pickle (optional)", type=['pkl'])
    if upload_new is not None:
        newdf = pd.read_csv(upload_new)
        st.write("Preview of uploaded dataset:")
        st.dataframe(newdf.head())
        # If models in session state, use them; else require model pickle
        if 'models' in st.session_state and 'preprocessor' in st.session_state and 'label_encoder' in st.session_state:
            models = st.session_state['models']
            preprocessor = st.session_state['preprocessor']
            le = st.session_state['label_encoder']
            sel_feats = st.session_state['selected_features']
            if not set(sel_feats).issubset(set(newdf.columns)):
                st.warning("Uploaded data does not contain all features used for training. Attempting to use overlapping columns.")
            Xnew = newdf[sel_feats].copy()
            # apply preprocessor.transform (we need same columns order); handle missing cols
            try:
                Xnew_trans = preprocessor.transform(Xnew)
                # predict with chosen model (allow selection)
                model_choice = st.selectbox("Choose model for prediction", options=list(models.keys()))
                pred = models[model_choice].predict(Xnew_trans)
                pred_labels = st.session_state['label_encoder'].inverse_transform(pred)
                newdf['Predicted_Q38_Followup_Interest'] = pred_labels
                st.write(newdf.head())
                download_df(newdf, filename="predictions_with_label.csv")
            except Exception as e:
                st.error(f"Error during preprocessing/prediction: {e}")
        else:
            st.info("No trained models in session. You can upload a models pickle (dictionary with {name:estimator}, preprocessor, label_encoder)")
            if model_upload is not None:
                try:
                    obj = joblib.load(model_upload)
                    models = obj.get('models')
                    preprocessor = obj.get('preprocessor')
                    le = obj.get('label_encoder')
                    sel_feats = obj.get('selected_features')
                    Xnew = newdf[sel_feats].copy()
                    Xnew_trans = preprocessor.transform(Xnew)
                    model_choice = st.selectbox("Choose model for prediction", options=list(models.keys()))
                    pred = models[model_choice].predict(Xnew_trans)
                    pred_labels = le.inverse_transform(pred)
                    newdf['Predicted_Q38_Followup_Interest'] = pred_labels
                    st.write(newdf.head())
                    download_df(newdf, filename="predictions_with_label.csv")
                except Exception as e:
                    st.error(f"Could not use uploaded model file: {e}")
    else:
        st.info("Upload a dataset to generate predictions. Alternatively train models in Model Trainer tab and then upload new data.")

st.markdown("---")
st.markdown("**Notes:** This app uses default package versions (no pins) to reduce Streamlit Cloud compatibility issues. For production, consider more robust validation, SHAP explainability, and model persistence.")
