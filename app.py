import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Pipeline",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 800; }
  .metric-lbl { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.1em; }
  .predict-result {
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
  }
  .survived { background: rgba(0,196,140,0.15); border: 1px solid rgba(0,196,140,0.4); color: #00c48c; }
  .not-survived { background: rgba(255,79,123,0.15); border: 1px solid rgba(255,79,123,0.4); color: #ff4f7b; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
DATA_PATH  = "Data/train.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    # fallback sample data for Streamlit Cloud demo
    st.warning("train.csv not found — using built-in sample data.")
    return pd.DataFrame({
        'Survived': [0,1,1,0,1,0,1,0,0,1]*89,
        'Pclass':   [3,1,2,3,1,2,3,1,2,3]*89,
        'Sex':      ['male','female','female','male','female','male','female','male','male','female']*89,
        'Age':      [22,38,26,35,35,np.nan,54,2,27,14]*89,
        'SibSp':    [1,1,0,1,0,0,0,3,0,1]*89,
        'Parch':    [0,0,0,0,0,0,0,1,2,0]*89,
        'Fare':     [7.25,71.28,7.93,53.1,8.05,8.46,51.86,21.08,11.13,30.07]*89,
    })

def preprocess(df):
    d = df.copy()
    d['Age'].fillna(d['Age'].median(), inplace=True)
    d['Fare'].fillna(d['Fare'].median(), inplace=True)
    if d['Sex'].dtype == object:
        d['Sex'] = d['Sex'].map({'male': 1, 'female': 0})
    return d

def build_and_train(df):
    d = preprocess(df)
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
    X = d[features]
    y = d['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(pipeline, X, y, cv=5)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    fi  = pipeline.named_steps['model'].feature_importances_
    return pipeline, acc, cv, cm, report, fi, features

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg", use_column_width=True)
st.sidebar.title("🚢 Titanic ML Pipeline")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🎯 Predict", "📊 EDA", "🔁 Train Model", "📦 Batch Predict", "📋 Evaluate"])

# ── Load data once ────────────────────────────────────────────────────────────
df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚢 Titanic Survival — ML Pipeline")
    st.markdown("End-to-end machine learning pipeline: data → preprocessing → training → prediction.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Passengers", len(df))
    with c2:
        survived = df['Survived'].sum()
        st.metric("Survived", int(survived))
    with c3:
        rate = df['Survived'].mean() * 100
        st.metric("Survival Rate", f"{rate:.1f}%")
    with c4:
        model = load_model()
        st.metric("Model Status", "✅ Ready" if model else "⚠️ Not Trained")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.code("""
Data (train.csv)
   ↓
Preprocessing  →  Fill missing Age/Fare  →  Encode Sex (male=1, female=0)
   ↓
sklearn Pipeline  →  SimpleImputer  →  StandardScaler  →  RandomForestClassifier
   ↓
Trained Model  →  models/model.pkl
   ↓
Predictions  →  Single / Batch CSV
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.title("🎯 Single Passenger Prediction")
    model = load_model()
    if model is None:
        st.error("No trained model found. Go to **🔁 Train Model** first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex    = st.selectbox("Sex", ["male", "female"])
        with c2:
            age    = st.slider("Age", 1, 80, 28)
            fare   = st.slider("Fare (£)", 0, 300, 15)
        with c3:
            sibsp  = st.number_input("Siblings/Spouse", 0, 8, 0)
            parch  = st.number_input("Parents/Children", 0, 6, 0)

        if st.button("⚡ Predict Survival", use_container_width=True):
            sex_enc = 1 if sex == "male" else 0
            inp = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare]],
                               columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])
            pred = model.predict(inp)[0]
            prob = model.predict_proba(inp)[0]

            if pred == 1:
                st.markdown('<div class="predict-result survived">✅ SURVIVED — Probability: {:.1f}%</div>'.format(prob[1]*100), unsafe_allow_html=True)
            else:
                st.markdown('<div class="predict-result not-survived">❌ DID NOT SURVIVE — Probability: {:.1f}%</div>'.format(prob[0]*100), unsafe_allow_html=True)

            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.subheader("Input Summary")
                st.json({"Pclass": pclass, "Sex": sex, "Age": age,
                         "SibSp": int(sibsp), "Parch": int(parch), "Fare": fare})
            with cc2:
                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({"Outcome": ["Did Not Survive","Survived"],
                                        "Probability": [prob[0]*100, prob[1]*100]})
                st.bar_chart(prob_df.set_index("Outcome"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Survival Count")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x='Survived', data=df, palette=['#ff4f7b','#00c48c'], ax=ax)
        ax.set_xticklabels(['Not Survived','Survived'])
        ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white'); ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        st.pyplot(fig)

    with c2:
        st.subheader("Survival by Class")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x='Pclass', hue='Survived', data=df, palette=['#ff4f7b','#00c48c'], ax=ax)
        ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white'); ax.legend(labels=['No','Yes'], facecolor='#1a1f2e', labelcolor='white')
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        df['Age'].dropna().hist(bins=25, ax=ax, color='#00e5ff', edgecolor='#0a0e1a')
        ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white')
        st.pyplot(fig)

    with c4:
        st.subheader("Survival by Sex")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(x='Sex', hue='Survived', data=df, palette=['#ff4f7b','#00c48c'], ax=ax)
        ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white'); ax.legend(labels=['No','Yes'], facecolor='#1a1f2e', labelcolor='white')
        st.pyplot(fig)

    st.subheader("Fare vs Age (coloured by Survival)")
    fig, ax = plt.subplots(figsize=(10,4))
    colors = df['Survived'].map({0:'#ff4f7b', 1:'#00c48c'})
    ax.scatter(df['Age'], df['Fare'], c=colors, alpha=0.5, s=20)
    ax.set_xlabel("Age", color='white'); ax.set_ylabel("Fare", color='white')
    ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
    ax.tick_params(colors='white')
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Train
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Train Model":
    st.title("🔁 Train the Model")
    st.info("Click the button below to train a Random Forest on the Titanic dataset and save `models/model.pkl`.")

    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("Training in progress..."):
            pipeline, acc, cv, cm, report, fi, features = build_and_train(df)

        st.success(f"✅ Model trained and saved! Accuracy: **{acc*100:.2f}%**")
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc*100:.1f}%")
        c2.metric("CV Mean",   f"{cv.mean()*100:.1f}%")
        c3.metric("CV Std",    f"±{cv.std()*100:.1f}%")
        c4.metric("Estimators","100")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No','Yes'], yticklabels=['No','Yes'])
            ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
            ax.tick_params(colors='white')
            st.pyplot(fig)

        with col2:
            st.subheader("Feature Importance")
            fi_df = pd.DataFrame({'Feature': features, 'Importance': fi}).sort_values('Importance')
            fig, ax = plt.subplots(figsize=(4,3))
            ax.barh(fi_df['Feature'], fi_df['Importance'], color='#00e5ff')
            ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
            ax.tick_params(colors='white')
            st.pyplot(fig)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Batch Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Predict":
    st.title("📦 Batch Predictions")
    model = load_model()
    if model is None:
        st.error("No trained model found. Go to **🔁 Train Model** first.")
    else:
        st.markdown("Upload a CSV with columns: `Pclass, Sex, Age, SibSp, Parch, Fare`")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("⚡ Run Batch Predictions"):
                d = preprocess(batch_df)
                features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
                d['Prediction'] = model.predict(d[features])
                d['Survived_Label'] = d['Prediction'].map({0:'No', 1:'Yes'})
                d['Survival_Probability'] = model.predict_proba(d[features])[:,1].round(3)

                st.success(f"✅ {len(d)} predictions complete!")
                st.dataframe(d, use_container_width=True)

                csv = d.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions CSV", csv,
                                   "predictions.csv", "text/csv", use_container_width=True)
        else:
            st.info("No file uploaded yet — showing predictions on training data as demo.")
            d = preprocess(df.copy())
            features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
            d['Prediction'] = model.predict(d[features])
            d['Survived_Label'] = d['Prediction'].map({0:'No', 1:'Yes'})
            st.dataframe(d[['Pclass','Sex','Age','Fare','Survived','Prediction','Survived_Label']].head(20),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Evaluate":
    st.title("📋 Model Evaluation")
    model = load_model()
    if model is None:
        st.error("No trained model found. Go to **🔁 Train Model** first.")
    else:
        d = preprocess(df)
        features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
        X = d[features]; y = d['Survived']

        with st.spinner("Running 5-fold cross validation..."):
            cv_scores = cross_val_score(model, X, y, cv=5)

        st.subheader("5-Fold Cross Validation")
        c1,c2,c3 = st.columns(3)
        c1.metric("Mean Accuracy", f"{cv_scores.mean()*100:.2f}%")
        c2.metric("Std Deviation", f"±{cv_scores.std()*100:.2f}%")
        c3.metric("Min / Max", f"{cv_scores.min()*100:.1f}% / {cv_scores.max()*100:.1f}%")

        fig, ax = plt.subplots(figsize=(7,3))
        ax.bar(range(1,6), cv_scores*100, color='#00e5ff', edgecolor='#0a0e1a')
        ax.axhline(cv_scores.mean()*100, color='#ff4f7b', linestyle='--', label=f'Mean {cv_scores.mean()*100:.1f}%')
        ax.set_xlabel("Fold", color='white'); ax.set_ylabel("Accuracy %", color='white')
        ax.set_facecolor('#1a1f2e'); fig.patch.set_facecolor('#1a1f2e')
        ax.tick_params(colors='white'); ax.legend(facecolor='#1a1f2e', labelcolor='white')
        st.pyplot(fig)

        st.subheader("Full Classification Report")
        y_pred = model.predict(X)
        report = classification_report(d['Survived'], y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)