import streamlit as st
import joblib
import re
import os
import numpy as np
import nltk
import cv2
import mediapipe as mp

from streamlit_mic_recorder import speech_to_text

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Devotional AI Assistant",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLEAN COMPACT UI
# ============================================================

st.markdown(
    """
    <style>

    /* REMOVE EXTRA TOP AND BOTTOM SPACE */

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
    }

    footer {
        visibility: hidden !important;
        height: 0 !important;
    }


    /* MAIN BACKGROUND */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(156, 39, 176, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 85%,
                rgba(103, 58, 183, 0.14),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f7f0ff 0%,
                #fdfaff 50%,
                #eee4ff 100%
            );

        color: #2d1838;
    }


    /* GENERAL TEXT */

    .stApp p,
    .stApp label {
        color: #2d1838;
    }


    h1 {
        color: #5e35b1 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #673ab7 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #7b1fa2 !important;
        font-weight: 700 !important;
    }


    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #311b5e 0%,
                #4527a0 50%,
                #1f123d 100%
            ) !important;

        border-right: 2px solid #b39ddb;
    }

    section[data-testid="stSidebar"] * {
        color: #f5edff !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e1bee7 !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        color: #f5edff !important;
        font-weight: 600 !important;
    }


    /* TEXT AREA */

    .stTextArea textarea {
        background-color: #fdfaff !important;
        color: #291535 !important;
        border: 2px solid #b39ddb !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    .stTextArea textarea::placeholder {
        color: #806b8a !important;
        opacity: 1 !important;
    }

    .stTextArea textarea:focus {
        border: 2px solid #7e57c2 !important;

        box-shadow:
            0 0 0 3px
            rgba(126, 87, 194, 0.14) !important;
    }


    /* TEXT INPUT */

    .stTextInput input {
        background-color: #fdfaff !important;
        color: #291535 !important;
        border: 2px solid #b39ddb !important;
        border-radius: 10px !important;
    }


    /* BUTTONS */

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #7b1fa2,
                #4527a0
            ) !important;

        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        padding: 8px 16px !important;

        box-shadow:
            0 3px 10px
            rgba(81, 45, 168, 0.18);
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #9c27b0,
                #5e35b1
            ) !important;

        color: #ffffff !important;
    }

    .stButton > button p {
        color: #ffffff !important;
    }


    /* SLIDER */

    .stSlider label {
        color: #5e35b1 !important;
        font-weight: 650 !important;
    }


    /* RADIO */

    div[role="radiogroup"] label {
        color: #2d1838 !important;
        font-weight: 600 !important;
    }


    /* ALERTS */

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    div[data-testid="stAlert"] p {
        color: #291535 !important;
    }


    /* METRICS */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #fdfaff,
                #f1e8ff
            );

        border: 1px solid #c5b3e6;
        border-radius: 12px;
        padding: 12px;

        box-shadow:
            0 3px 10px
            rgba(81, 45, 168, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #654d73 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #5e35b1 !important;
    }


    /* EXPANDER */

    div[data-testid="stExpander"] {
        background: #fdfaff !important;
        border: 1px solid #c5b3e6 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #5e35b1 !important;
        font-weight: 700 !important;
    }


    /* FILE UPLOADER */

    section[data-testid="stFileUploaderDropzone"] {
        background: #fdfaff !important;
        border: 2px dashed #9575cd !important;
        border-radius: 12px !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: #4a3655 !important;
    }


    /* IMAGES */

    img {
        border-radius: 10px;
    }


    /* DIVIDER */

    hr {
        margin-top: 0.8rem !important;
        margin-bottom: 0.8rem !important;

        border-color:
            rgba(103, 58, 183, 0.20) !important;
    }


    /* REDUCE VERTICAL SPACE */

    .stMarkdown {
        margin-bottom: 0.3rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "intent_text" not in st.session_state:
    st.session_state.intent_text = ""

if "ramayana_query" not in st.session_state:
    st.session_state.ramayana_query = ""


# ============================================================
# NLTK RESOURCES
# ============================================================

@st.cache_resource
def download_nltk_resources():

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet")
    ]

    for path, name in resources:

        try:
            nltk.data.find(path)

        except LookupError:

            nltk.download(
                name,
                quiet=True
            )


download_nltk_resources()


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# LOAD INTENT MODELS
# ============================================================

@st.cache_resource
def load_intent_models():

    model = joblib.load(
        os.path.join(
            BASE_DIR,
            "best_model_tfidf_w2v.pkl"
        )
    )

    tfidf = joblib.load(
        os.path.join(
            BASE_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    word2vec_model = KeyedVectors.load(
        os.path.join(
            BASE_DIR,
            "word2vec_vectors.kv"
        )
    )

    class_labels = joblib.load(
        os.path.join(
            BASE_DIR,
            "class_labels.pkl"
        )
    )

    return (
        model,
        tfidf,
        word2vec_model,
        class_labels
    )


(
    model,
    tfidf,
    word2vec_model,
    class_labels
) = load_intent_models()


# ============================================================
# LOAD RAMAYANA
# ============================================================

@st.cache_resource
def load_ramayana():

    ramayana_data = joblib.load(
        os.path.join(
            BASE_DIR,
            "ramayana_data.pkl"
        )
    )

    ramayana_tfidf_matrix = joblib.load(
        os.path.join(
            BASE_DIR,
            "ramayana_tfidf_matrix.pkl"
        )
    )

    ramayana_tfidf_vectorizer = joblib.load(
        os.path.join(
            BASE_DIR,
            "ramayana_tfidf_vectorizer.pkl"
        )
    )

    return (
        ramayana_data,
        ramayana_tfidf_matrix,
        ramayana_tfidf_vectorizer
    )


(
    ramayana_data,
    ramayana_tfidf_matrix,
    ramayana_tfidf_vectorizer
) = load_ramayana()


# ============================================================
# LOAD MUDRA MODELS
# ============================================================

@st.cache_resource
def load_mudra_models():

    mudra_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "mudra_svm_model.pkl"
        )
    )

    mudra_scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "mudra_scaler.pkl"
        )
    )

    mudra_label_encoder = joblib.load(
        os.path.join(
            BASE_DIR,
            "mudra_label_encoder.pkl"
        )
    )

    mudra_info = joblib.load(
        os.path.join(
            BASE_DIR,
            "mudra_info.pkl"
        )
    )

    return (
        mudra_model,
        mudra_scaler,
        mudra_label_encoder,
        mudra_info
    )


(
    mudra_model,
    mudra_scaler,
    mudra_label_encoder,
    mudra_info
) = load_mudra_models()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

HAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)


@st.cache_resource
def load_hand_landmarker():

    if not os.path.exists(
        HAND_MODEL_PATH
    ):
        return None

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),

        running_mode=VisionRunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(
        options
    )


hand_landmarker = load_hand_landmarker()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(
        text
    )

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# WORD2VEC SENTENCE VECTOR
# ============================================================

def sentence_vector(sentence):

    words = sentence.split()

    vectors = [

        word2vec_model[word]

        for word in words

        if word in word2vec_model
    ]

    if len(vectors) == 0:

        return np.zeros(
            word2vec_model.vector_size
        )

    vectors = np.array(
        vectors
    )

    return np.mean(
        vectors,
        axis=0
    )


# ============================================================
# INTENT PREDICTION
# ============================================================

def predict_intent(text):

    processed_text = preprocess_text(
        text
    )

    w2v_vector = sentence_vector(
        processed_text
    )

    w2v_vector = w2v_vector.reshape(
        1,
        -1
    )

    tfidf_vector = tfidf.transform(
        [processed_text]
    ).toarray()

    combined_vector = np.hstack([
        w2v_vector,
        tfidf_vector
    ])

    expected_features = (
        model.n_features_in_
    )

    actual_features = (
        combined_vector.shape[1]
    )

    if actual_features != expected_features:

        raise ValueError(
            f"Feature mismatch: "
            f"model expects "
            f"{expected_features}, "
            f"received "
            f"{actual_features}"
        )

    prediction_result = model.predict(
        combined_vector
    )[0]

    if isinstance(
        prediction_result,
        (int, np.integer)
    ):

        prediction = class_labels[
            prediction_result
        ]

    else:

        prediction = prediction_result

    probabilities = model.predict_proba(
        combined_vector
    )[0]

    probability = (
        probabilities.max() * 100
    )

    return (
        processed_text,
        prediction,
        probability
    )


# ============================================================
# RAMAYANA SEARCH
# ============================================================

def search_ramayana(
    query,
    top_k=3
):

    query_vector = (
        ramayana_tfidf_vectorizer.transform(
            [query]
        )
    )

    similarities = cosine_similarity(
        query_vector,
        ramayana_tfidf_matrix
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        row = ramayana_data.iloc[
            index
        ]

        results.append({

            "kanda": row["kanda"],

            "content": row["content"],

            "explanation": row["explanation"],

            "similarity": similarities[index]

        })

    return results


# ============================================================
# MEDIAPIPE LANDMARK EXTRACTION
# ============================================================

def extract_hand_landmarks(image):

    if hand_landmarker is None:

        return None

    if len(image.shape) == 2:

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    else:

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = hand_landmarker.detect(
        mp_image
    )

    if not result.hand_landmarks:

        return None

    hand = result.hand_landmarks[0]

    landmarks = []

    for landmark in hand:

        landmarks.extend([

            landmark.x,
            landmark.y,
            landmark.z

        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    landmarks = landmarks.reshape(
        21,
        3
    )

    wrist = landmarks[0].copy()

    landmarks = (
        landmarks - wrist
    )

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    max_distance = np.max(
        distances
    )

    if max_distance > 0:

        landmarks = (
            landmarks /
            max_distance
        )

    return landmarks.flatten()


# ============================================================
# MUDRA PREDICTION
# ============================================================

def predict_mudra(image):

    landmarks = extract_hand_landmarks(
        image
    )

    if landmarks is None:

        return None, None

    landmarks = landmarks.reshape(
        1,
        -1
    )

    scaled_landmarks = (
        mudra_scaler.transform(
            landmarks
        )
    )

    prediction = mudra_model.predict(
        scaled_landmarks
    )[0]

    mudra_name = (
        mudra_label_encoder.inverse_transform(
            [prediction]
        )[0]
    )

    probabilities = (
        mudra_model.predict_proba(
            scaled_landmarks
        )[0]
    )

    probability = (
        probabilities.max() * 100
    )

    return (
        mudra_name,
        probability
    )


# ============================================================
# SIMPLE PLAIN TITLE
# ============================================================

st.title("🙏 Devotional AI Assistant")

st.write(
    "Discover wisdom through AI, devotion and Indian spiritual heritage"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        font-size:23px;
        font-weight:800;
        color:#e1bee7 !important;
        padding:4px 0 8px 0;
    ">
        🕉️ Devotional AI
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "🪔 Navigation",
    [
        "💬 Intent Detection",
        "📜 Ramayana",
        "🤲 Mudra Recognition"
    ]
)


# ============================================================
# INTENT DETECTION
# ============================================================

if page == "💬 Intent Detection":

    st.header(
        "💬 Devotional Intent Detection"
    )

    st.write(
        "Enter a devotional question or use your microphone."
    )

    st.subheader(
        "🎤 Speech-to-Text"
    )

    spoken_text = speech_to_text(
        language="en",
        start_prompt="🎤 Start Speaking",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="intent_speech"
    )

    if spoken_text:

        st.session_state.intent_text = (
            spoken_text
        )

        st.success(
            "Speech recognized successfully."
        )

        st.write(
            f"**Recognized Speech:** {spoken_text}"
        )

    text_input = st.text_area(
        "Enter your question:",
        placeholder="Example: How can I control my mind?",
        key="intent_text_area",
        height=110
    )

    if st.button(
        "🔮 Predict Intent",
        key="intent_button"
    ):

        final_text = (
            st.session_state.intent_text
            if st.session_state.intent_text.strip()
            else text_input
        )

        if not final_text.strip():

            st.warning(
                "Please enter a question or use the microphone."
            )

        else:

            try:

                (
                    processed_text,
                    prediction,
                    probability
                ) = predict_intent(
                    final_text
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "Recognized Text"
                    )

                    st.write(
                        final_text
                    )

                with col2:

                    st.subheader(
                        "Processed Text"
                    )

                    st.write(
                        processed_text
                    )

                st.subheader(
                    "🔮 Predicted Intent"
                )

                st.success(
                    prediction
                )

                st.subheader(
                    "🎯 Prediction Probability"
                )

                st.progress(
                    min(
                        probability / 100,
                        1.0
                    )
                )

                st.write(
                    f"{probability:.2f}%"
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# RAMAYANA
# ============================================================

elif page == "📜 Ramayana":

    st.header(
        "📜 Ramayana Verse Search"
    )

    st.write(
        "Ask a question about the Ramayana. "
        "You can type your question or speak it."
    )

    st.subheader(
        "🎤 Ask by Voice"
    )

    ramayana_spoken_text = speech_to_text(
        language="en",
        start_prompt="🎤 Speak Ramayana Question",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="ramayana_speech"
    )

    if ramayana_spoken_text:

        st.session_state.ramayana_query = (
            ramayana_spoken_text
        )

        st.success(
            "Speech recognized successfully."
        )

        st.write(
            f"**Recognized Speech:** {ramayana_spoken_text}"
        )

    query = st.text_area(
        "Enter your Ramayana question:",
        placeholder="Example: What did Hanuman do in Lanka?",
        key="ramayana_query_area",
        height=110
    )

    number_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button(
        "🔎 Search Ramayana",
        key="ramayana_button"
    ):

        final_query = (
            st.session_state.ramayana_query
            if st.session_state.ramayana_query.strip()
            else query
        )

        if not final_query.strip():

            st.warning(
                "Please enter a question or use the microphone."
            )

        else:

            try:

                results = search_ramayana(
                    final_query,
                    number_results
                )

                st.success(
                    f"Found {len(results)} relevant verses."
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.divider()

                    st.subheader(
                        f"📜 Result {i}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**📖 Kanda:** "
                            f"{result['kanda']}"
                        )

                    with col2:

                        st.write(
                            f"**🎯 Similarity:** "
                            f"{result['similarity']:.4f}"
                        )

                    st.markdown(
                        "**🕉️ Sanskrit Verse:**"
                    )

                    st.info(
                        result["content"]
                    )

                    st.markdown(
                        "**📖 Explanation:**"
                    )

                    st.write(
                        result["explanation"]
                    )

            except Exception as e:

                st.error(
                    "Ramayana search failed."
                )

                st.exception(e)


# ============================================================
# MUDRA RECOGNITION
# ============================================================

elif page == "🤲 Mudra Recognition":

    st.header(
        "🤲 Indian Mudra Recognition"
    )

    st.write(
        "Use your camera or upload an image of a hand Mudra."
    )

    if hand_landmarker is None:

        st.error(
            "MediaPipe hand_landmarker.task was not found."
        )

        st.info(
            "Place hand_landmarker.task in the same folder as app.py."
        )

    else:

        input_method = st.radio(
            "Choose input method:",
            [
                "📷 Camera",
                "🖼️ Upload Image"
            ],
            horizontal=True
        )

        image = None

        if input_method == "📷 Camera":

            camera_image = st.camera_input(
                "Take a picture of your Mudra"
            )

            if camera_image is not None:

                file_bytes = np.asarray(
                    bytearray(
                        camera_image.read()
                    ),
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

        else:

            uploaded_file = st.file_uploader(
                "Upload a Mudra image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            if uploaded_file is not None:

                file_bytes = np.asarray(
                    bytearray(
                        uploaded_file.read()
                    ),
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_COLOR
                )

        if image is not None:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Input Mudra",
                use_container_width=True
            )

            if st.button(
                "🤲 Recognize Mudra",
                key="mudra_button"
            ):

                try:

                    (
                        mudra_name,
                        probability
                    ) = predict_mudra(
                        image
                    )

                    if mudra_name is None:

                        st.error(
                            "No hand was detected in the image."
                        )

                    else:

                        col1, col2 = st.columns(2)

                        with col1:

                            st.subheader(
                                "🔮 Predicted Mudra"
                            )

                            st.success(
                                mudra_name
                            )

                        with col2:

                            st.subheader(
                                "🎯 Probability"
                            )

                            st.write(
                                f"{probability:.2f}%"
                            )

                            st.progress(
                                min(
                                    probability / 100,
                                    1.0
                                )
                            )

                        if mudra_name in mudra_info:

                            info = mudra_info[
                                mudra_name
                            ]

                            col1, col2 = st.columns(2)

                            with col1:

                                st.subheader(
                                    "🌸 Meaning"
                                )

                                st.write(
                                    info.get(
                                        "meaning",
                                        "Information unavailable."
                                    )
                                )

                            with col2:

                                st.subheader(
                                    "📖 Significance"
                                )

                                st.write(
                                    info.get(
                                        "significance",
                                        "Information unavailable."
                                    )
                                )

                except Exception as e:

                    st.error(
                        "Mudra recognition failed."
                    )

                    st.exception(e)


# ============================================================
# COMPACT FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6c5878;
        font-size:13px;
        padding:5px 0 0 0;
    ">
        🙏 Devotional AI Assistant
        &nbsp; | &nbsp;
        🧠 NLP + Word2Vec + TF-IDF
        &nbsp; | &nbsp;
        🎤 Speech-to-Text
        &nbsp; | &nbsp;
        📜 Ramayana Retrieval
        &nbsp; | &nbsp;
        🤲 MediaPipe Mudra Recognition
    </div>
    """,
    unsafe_allow_html=True
)
