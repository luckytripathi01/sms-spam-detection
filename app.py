import streamlit as st
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ---------------- TEXT TRANSFORM ----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

header[data-testid="stHeader"] {background: transparent;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #5f72ff 0%,
        #7a5fff 40%,
        #9b5de5 75%,
        #c77dff 100%
    );
    background-attachment: fixed;
}

/* SIDEBAR COLOR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #5f72ff 0%,
        #7a5fff 50%,
        #9b5de5 100%
    );
    padding: 20px;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* HERO BOX */
.hero {
    background: rgba(255,255,255,0.18);
    padding: 35px;
    border-radius: 25px;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    text-align: center;
    margin-bottom: 30px;
    color: white;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

/* SIDEBAR CARD */
.sidebar-card {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 25px;
    backdrop-filter: blur(15px);
    margin-bottom: 25px;
}

.history-item {
    background: rgba(255,255,255,0.20);
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🤖 AI SMS Spam Detector</h1>
    <p>Smart AI Powered Spam Classification</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
message = st.text_area("✍️ Enter your SMS message")

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze Message"):

    if message:

        transformed_message = transform_text(message)
        vector_input = vectorizer.transform([transformed_message])

        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]
        confidence = round(np.max(probability) * 100, 2)

        result_text = "SPAM" if prediction == 1 else "NOT SPAM"

        # Save history
        st.session_state.history.append({
            "message": message,
            "result": result_text,
            "confidence": confidence
        })

        if prediction == 1:
            st.markdown(f"""
            <div class="result" style="background: linear-gradient(90deg,#ff4b2b,#ff416c); color:white;">
            🚨 SPAM DETECTED <br><br>
            Confidence Score: {confidence}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result" style="background: linear-gradient(90deg,#00b09b,#96c93d); color:white;">
            ✅ NOT SPAM <br><br>
            Confidence Score: {confidence}%
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Please enter a message first.")

# ---------------- SIDEBAR ----------------
# -------------- SIDEBAR --------------

with st.sidebar:

    # MODEL INFO BOX
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Information")
    st.write("• Algorithm: Multinomial Naive Bayes")
    st.write("• Vectorizer: TF-IDF")
    st.write("• Dataset: SMS Spam Dataset")
    st.markdown("</div>", unsafe_allow_html=True)

    # HISTORY BOX
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Prediction History")

    if st.session_state.history:

        for item in reversed(st.session_state.history):

            st.markdown(f"""
            <div class="history-item">
            <b>{item['result']}</b> ({item['confidence']}%)<br>
            {item['message'][:60]}...
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.write("No predictions yet.")

    st.markdown("</div>", unsafe_allow_html=True)