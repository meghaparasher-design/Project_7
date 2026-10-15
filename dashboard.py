from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_persistence import load_model
from src.prediction import predict_tweet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Disaster Tweet AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "disaster_tweet_svm_pipeline.pkl"
)

FINAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_tuned_model_results.csv"
)

VALIDATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tuned_model_validation.csv"
)

FALSE_POSITIVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "false_positives.csv"
)

FALSE_NEGATIVE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "false_negatives.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                135deg,
                #06111f 0%,
                #0b1730 45%,
                #111827 100%
            );
    }

    .block-container {
        max-width: 1350px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07111f,
                #101b35
            );

        border-right:
            1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    .hero {
        padding: 42px 35px;
        border-radius: 25px;
        text-align: center;

        background:
            linear-gradient(
                120deg,
                #2563EB,
                #7C3AED,
                #DB2777
            );

        box-shadow:
            0px 15px 45px rgba(0,0,0,0.35);

        margin-bottom: 25px;
    }

    .hero-title {
        color: white;
        font-size: 46px;
        font-weight: 800;
    }

    .hero-subtitle {
        color: #F1F5F9;
        font-size: 18px;
        margin-top: 5px;
    }

    .badge {
        display: inline-block;

        padding: 6px 13px;
        margin: 4px;

        border-radius: 20px;

        background:
            rgba(255,255,255,0.14);

        border:
            1px solid rgba(255,255,255,0.25);

        color: white;

        font-size: 13px;
        font-weight: 600;
    }

    .info-card {
        padding: 22px;

        border-radius: 18px;

        background:
            rgba(15,23,42,0.85);

        border:
            1px solid rgba(129,140,248,0.22);

        box-shadow:
            0px 8px 25px rgba(0,0,0,0.20);

        margin-bottom: 18px;
    }

    .disaster-box {
        padding: 30px;

        border-radius: 22px;

        text-align: center;

        background:
            linear-gradient(
                135deg,
                #7F1D1D,
                #991B1B
            );

        border:
            1px solid #EF4444;

        box-shadow:
            0px 10px 30px rgba(239,68,68,0.20);
    }

    .safe-box {
        padding: 30px;

        border-radius: 22px;

        text-align: center;

        background:
            linear-gradient(
                135deg,
                #064E3B,
                #047857
            );

        border:
            1px solid #10B981;

        box-shadow:
            0px 10px 30px rgba(16,185,129,0.20);
    }

    .result-title {
        color: white;
        font-size: 30px;
        font-weight: 800;
    }

    .result-text {
        color: #E2E8F0;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.96),
                rgba(15,23,42,0.96)
            );

        border:
            1px solid rgba(129,140,248,0.25);

        border-radius: 18px;

        padding: 17px;

        box-shadow:
            0px 7px 22px rgba(0,0,0,0.20);
    }

    div[data-testid="stMetricValue"] {
        color: #A5B4FC;
        font-weight: 800;
    }

    .stTextArea textarea {
        background: #0F172A !important;

        color: white !important;

        border:
            1px solid rgba(129,140,248,0.45) !important;

        border-radius: 15px !important;

        font-size: 16px !important;
    }

    div.stButton > button {
        width: 100%;

        border: none;

        border-radius: 13px;

        padding: 0.75rem;

        font-weight: 700;

        color: white;

        background:
            linear-gradient(
                90deg,
                #2563EB,
                #7C3AED,
                #DB2777
            );

        transition: 0.2s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0px 8px 25px rgba(124,58,237,0.40);

        color: white;
    }

    .footer {
        text-align: center;

        color: #94A3B8;

        padding-top: 35px;

        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CACHED DATA / MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():

    return load_model(
        MODEL_PATH
    )


@st.cache_data
def load_csv_if_exists(path):

    if path.exists():
        return pd.read_csv(path)

    return None


try:

    model = get_model()

except Exception as error:

    st.error(
        f"Unable to load model: {error}"
    )

    st.stop()


final_results = load_csv_if_exists(
    FINAL_RESULTS_PATH
)

validation_results = load_csv_if_exists(
    VALIDATION_RESULTS_PATH
)

false_positives = load_csv_if_exists(
    FALSE_POSITIVE_PATH
)

false_negatives = load_csv_if_exists(
    FALSE_NEGATIVE_PATH
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🚨 Disaster Tweet AI"
    )

    st.caption(
        "Machine Learning Dashboard"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Live Prediction",
            "📊 Model Performance",
            "🧩 Error Analysis",
            "🧠 Project Details"
        ]
    )

    st.divider()

    st.markdown(
        "### 🏆 Final Model"
    )

    st.success(
        "Tuned Linear SVM"
    )

    st.write(
        "**TF-IDF:** Unigrams + Bigrams"
    )

    st.write(
        "**C:** 0.1"
    )

    st.write(
        "**Class Weight:** Balanced"
    )

    st.divider()

    st.caption(
        "NLP Disaster Tweet Classification"
    )


# ============================================================
# REUSABLE HERO
# ============================================================

def show_hero():

    st.markdown(
        """
        <div class="hero">

            <div>
                <span class="badge">MACHINE LEARNING</span>
                <span class="badge">NLP</span>
                <span class="badge">REAL-TIME PREDICTION</span>
            </div>

            <div class="hero-title">
                🚨 Disaster Tweet AI
            </div>

            <div class="hero-subtitle">
                Intelligent Disaster Tweet Classification
                using Natural Language Processing
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    show_hero()

    st.markdown(
        """
        <div class="info-card">

        <h2>🌐 Project Overview</h2>

        <p>
        During emergencies, social media can provide
        valuable real-time information. However, words
        associated with disasters may also appear in jokes,
        metaphors, entertainment or everyday conversations.
        </p>

        <p>
        This project uses <b>Natural Language Processing</b>
        and <b>Machine Learning</b> to classify tweets into
        <b>Disaster</b> and <b>Non-Disaster</b> categories.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "## 📊 Project at a Glance"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Tweets",
            "7,613"
        )

    with c2:
        st.metric(
            "Final Accuracy",
            "81.09%"
        )

    with c3:
        st.metric(
            "Final F1 Score",
            "77.85%"
        )

    with c4:
        st.metric(
            "ROC-AUC",
            "87.09%"
        )


    st.markdown(
        "## ⚙️ ML Pipeline"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.info(
            """
            **1️⃣ Data Preparation**

            Missing values

            Text cleaning

            Lemmatization
            """
        )

    with p2:

        st.info(
            """
            **2️⃣ Feature Engineering**

            TF-IDF

            Unigrams

            Bigrams
            """
        )

    with p3:

        st.info(
            """
            **3️⃣ Model Training**

            Logistic Regression

            Naive Bayes

            Linear SVM
            """
        )

    with p4:

        st.success(
            """
            **4️⃣ Final Selection**

            Tuned

            Linear SVM

            F1 = 77.85%
            """
        )


# ============================================================
# LIVE PREDICTION PAGE
# ============================================================

elif page == "🔍 Live Prediction":

    st.title(
        "🔍 Real-Time Tweet Classification"
    )

    st.write(
        "Enter your own tweet or choose a sample below."
    )


    # --------------------------------------------------------
    # SAMPLE TWEETS
    # --------------------------------------------------------

    sample_option = st.selectbox(
        "Try a sample tweet",
        [
            "Write my own tweet",
            "Earthquake example",
            "Flood example",
            "Normal day example",
            "Movie example"
        ]
    )


    sample_tweets = {

        "Write my own tweet":
            "",

        "Earthquake example":
            "A major earthquake has damaged several buildings",

        "Flood example":
            "Emergency crews are responding to severe flooding",

        "Normal day example":
            "Had an amazing day at the beach with my friends",

        "Movie example":
            "That movie was an absolute disaster"
    }


    tweet = st.text_area(
        "Tweet",
        value=sample_tweets[
            sample_option
        ],
        height=160,
        max_chars=1000,
        placeholder="Type a tweet here..."
    )


    left, right = st.columns(
        [3, 1]
    )

    with left:

        st.caption(
            f"Characters: {len(tweet)} / 1000"
        )

    with right:

        st.caption(
            f"Words: {len(tweet.split())}"
        )


    analyze = st.button(
        "✨ Analyze Tweet",
        use_container_width=True
    )


    if analyze:

        if not tweet.strip():

            st.warning(
                "Please enter a tweet first."
            )

        else:

            try:

                with st.spinner(
                    "Running NLP analysis..."
                ):

                    result = predict_tweet(
                        model,
                        tweet
                    )


                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

                if result["prediction"] == 1:

                    st.markdown(
                        """
                        <div class="disaster-box">

                            <div class="result-title">
                                🚨 Disaster Tweet
                            </div>

                            <div class="result-text">
                                The model detected language
                                associated with a disaster
                                or emergency event.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="safe-box">

                            <div class="result-title">
                                ✅ Non-Disaster Tweet
                            </div>

                            <div class="result-text">
                                The model predicts that this
                                tweet does not describe a
                                disaster event.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.markdown(
                    "### 🔬 Live NLP Processing"
                )

                original_col, processed_col = (
                    st.columns(2)
                )

                with original_col:

                    st.markdown(
                        "**Original Tweet**"
                    )

                    st.info(
                        result[
                            "original_text"
                        ]
                    )

                with processed_col:

                    st.markdown(
                        "**Processed Tweet**"
                    )

                    st.code(
                        result[
                            "processed_text"
                        ]
                    )


                # --------------------------------------------
                # DECISION SCORE
                # --------------------------------------------

                processed_text = result[
                    "processed_text"
                ]

                if hasattr(
                    model,
                    "decision_function"
                ):

                    score = float(
                        model.decision_function(
                            [processed_text]
                        )[0]
                    )

                    st.markdown(
                        "### 📡 SVM Decision Analysis"
                    )

                    st.metric(
                        "Decision Score",
                        f"{score:.3f}"
                    )

                    if score > 0:

                        st.caption(
                            "A positive decision score places "
                            "the tweet on the Disaster side "
                            "of the SVM decision boundary."
                        )

                    else:

                        st.caption(
                            "A negative decision score places "
                            "the tweet on the Non-Disaster "
                            "side of the SVM decision boundary."
                        )

                    st.warning(
                        "The decision score is not a "
                        "probability or confidence percentage."
                    )


                with st.expander(
                    "🧠 How was this prediction produced?"
                ):

                    st.markdown(
                        """
                        **Step 1 — Cleaning**

                        URLs, mentions, symbols and other
                        unnecessary noise are removed.

                        **Step 2 — NLP Preprocessing**

                        Text is normalized, stopwords are
                        removed and words are lemmatized.

                        **Step 3 — TF-IDF**

                        The processed tweet is transformed into
                        numerical unigram and bigram features.

                        **Step 4 — Linear SVM**

                        The trained SVM determines which side
                        of its decision boundary the tweet
                        belongs to.
                        """
                    )


            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title(
        "📊 Model Performance Dashboard"
    )

    st.write(
        "Explore the performance of the models evaluated "
        "during the project."
    )


    # --------------------------------------------------------
    # FINAL MODEL METRICS
    # --------------------------------------------------------

    st.markdown(
        "### 🏆 Final Tuned Linear SVM"
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric(
            "Accuracy",
            "81.09%"
        )

    with m2:
        st.metric(
            "Precision",
            "78.33%"
        )

    with m3:
        st.metric(
            "Recall",
            "77.37%"
        )

    with m4:
        st.metric(
            "F1 Score",
            "77.85%"
        )

    with m5:
        st.metric(
            "ROC-AUC",
            "87.09%"
        )


    # --------------------------------------------------------
    # DYNAMIC FINAL RESULTS
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Tuned Model Comparison"
    )

    if final_results is not None:

        st.dataframe(
            final_results,
            use_container_width=True,
            hide_index=True
        )

        possible_metrics = [
            column
            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
            if column in final_results.columns
        ]

        if (
            "Model" in final_results.columns
            and possible_metrics
        ):

            metric = st.selectbox(
                "Select metric to compare",
                possible_metrics
            )

            chart_data = (
                final_results[
                    ["Model", metric]
                ]
                .set_index("Model")
            )

            st.bar_chart(
                chart_data
            )

    else:

        st.info(
            "final_tuned_model_results.csv "
            "was not found in the reports folder."
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    st.markdown(
        "### 🔁 Cross-Validation Results"
    )

    if validation_results is not None:

        st.dataframe(
            validation_results,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = (
            validation_results
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if (
            "Model" in validation_results.columns
            and numeric_columns
        ):

            selected_validation_metric = (
                st.selectbox(
                    "Select validation metric",
                    numeric_columns
                )
            )

            validation_chart = (
                validation_results[
                    [
                        "Model",
                        selected_validation_metric
                    ]
                ]
                .set_index("Model")
            )

            st.bar_chart(
                validation_chart
            )

    else:

        st.info(
            "tuned_model_validation.csv "
            "was not found."
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown(
        "### 🧩 Final Confusion Matrix"
    )

    cm1, cm2, cm3, cm4 = st.columns(4)

    with cm1:
        st.metric(
            "True Negatives",
            "729"
        )

    with cm2:
        st.metric(
            "False Positives",
            "140"
        )

    with cm3:
        st.metric(
            "False Negatives",
            "148"
        )

    with cm4:
        st.metric(
            "True Positives",
            "506"
        )


    confusion_data = pd.DataFrame(
        {
            "Predicted Non-Disaster":
                [729, 148],

            "Predicted Disaster":
                [140, 506]
        },
        index=[
            "Actual Non-Disaster",
            "Actual Disaster"
        ]
    )

    st.dataframe(
        confusion_data,
        use_container_width=True
    )


# ============================================================
# ERROR ANALYSIS PAGE
# ============================================================

elif page == "🧩 Error Analysis":

    st.title(
        "🧩 Model Error Analysis"
    )

    st.write(
        "Explore tweets that the final model "
        "classified incorrectly."
    )


    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "False Positives",
            "140"
        )

    with e2:
        st.metric(
            "False Negatives",
            "148"
        )

    with e3:
        st.metric(
            "Total Errors",
            "288"
        )


    error_type = st.radio(
        "Choose error type",
        [
            "False Positives",
            "False Negatives"
        ],
        horizontal=True
    )


    if error_type == "False Positives":

        st.markdown(
            "### ⚠️ False Positives"
        )

        st.write(
            """
            These tweets were actually Non-Disaster,
            but the model predicted Disaster.
            """
        )

        if false_positives is not None:

            number = st.slider(
                "Number of examples",
                5,
                min(
                    50,
                    len(false_positives)
                ),
                10
            )

            display_columns = [
                column
                for column in [
                    "text",
                    "processed_text",
                    "target",
                    "predicted_target"
                ]
                if column in false_positives.columns
            ]

            st.dataframe(
                false_positives[
                    display_columns
                ].head(number),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "false_positives.csv was not found."
            )


    else:

        st.markdown(
            "### ⚠️ False Negatives"
        )

        st.write(
            """
            These tweets were actually Disaster,
            but the model predicted Non-Disaster.
            """
        )

        if false_negatives is not None:

            number = st.slider(
                "Number of examples",
                5,
                min(
                    50,
                    len(false_negatives)
                ),
                10
            )

            display_columns = [
                column
                for column in [
                    "text",
                    "processed_text",
                    "target",
                    "predicted_target"
                ]
                if column in false_negatives.columns
            ]

            st.dataframe(
                false_negatives[
                    display_columns
                ].head(number),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "false_negatives.csv was not found."
            )


    st.markdown(
        "### 💡 Error Analysis Insight"
    )

    st.info(
        """
        False positives commonly occur when disaster-related
        vocabulary is used metaphorically, humorously or in
        entertainment and technical contexts.

        False negatives can occur when real disaster tweets use
        indirect wording, abbreviations, noisy text or uncommon
        expressions.

        This demonstrates one limitation of TF-IDF + Linear SVM:
        it learns statistical word and phrase patterns but has
        limited understanding of deeper semantic context.
        """
    )


# ============================================================
# PROJECT DETAILS PAGE
# ============================================================

elif page == "🧠 Project Details":

    st.title(
        "🧠 Project Architecture"
    )

    st.markdown(
        "### 🎯 Problem Statement"
    )

    st.write(
        """
        The objective of this project is to automatically
        determine whether a tweet describes a real disaster
        event or is unrelated to an actual disaster.
        """
    )


    st.markdown(
        "### 📦 Dataset"
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Total Tweets",
            "7,613"
        )

    with d2:
        st.metric(
            "Non-Disaster",
            "4,342"
        )

    with d3:
        st.metric(
            "Disaster",
            "3,271"
        )


    st.markdown(
        "### 🔬 Models Evaluated"
    )

    models_table = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Multinomial Naive Bayes",
                "Linear SVM",
                "Random Forest"
            ],

            "Role": [
                "Baseline linear classifier",
                "Probabilistic text classifier",
                "Final selected classifier",
                "Ensemble classifier"
            ]
        }
    )

    st.dataframe(
        models_table,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### 🏆 Final Model"
    )

    st.success(
        """
        **Tuned Linear SVM**

        Best parameters:

        • C = 0.1

        • class_weight = balanced

        • TF-IDF ngram_range = (1, 2)

        Final Holdout F1 Score = 77.85%
        """
    )


    st.markdown(
        "### 🧱 Modular Project Design"
    )

    st.code(
        """
Disaster_Tweet_Classification/

├── data/
├── notebooks/
├── models/
├── reports/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── validation.py
│   ├── model_persistence.py
│   └── prediction.py
│
└── dashboard.py
        """,
        language="text"
    )

    st.info(
        """
        The project follows modular programming principles.
        Data loading, preprocessing, feature engineering,
        training, evaluation, validation, model persistence
        and prediction are separated into reusable Python
        modules rather than keeping all logic inside the
        notebook.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>🚨 Disaster Tweet Classification</b>

        <br>

        NLP • Machine Learning • Linear SVM • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
