import os
import re
import faiss
import joblib
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =====================================================
# CONFIG
# =====================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SENTIMENT_HF_REPO = "mariamelkady/customer-support-sentiment"

GREETING_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s?(morning|afternoon|evening)|thanks|thank\s?you|"
    r"bye|goodbye|see\s?you|ok(ay)?|great|cool)\b[\s!.,]*$",
    re.IGNORECASE,
)

RAG_INTENTS = {"order_status", "order_management", "billing_and_refunds", "account_management"}


# =====================================================
# LOAD MODELS (cached so this only runs once per session)
# =====================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    lang_clf = joblib.load(f"{MODELS_DIR}/lang_classifier.pkl")
    lang_vectorizer = joblib.load(f"{MODELS_DIR}/lang_vectorizer.pkl")

    intent_clf = joblib.load(f"{MODELS_DIR}/intent_classifier.pkl")
    intent_vectorizer = joblib.load(f"{MODELS_DIR}/intent_vectorizer.pkl")

    sentiment_source = SENTIMENT_HF_REPO or f"{MODELS_DIR}/sentiment_model"
    sentiment_pipe = pipeline("text-classification", model=sentiment_source, tokenizer=sentiment_source)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(f"{MODELS_DIR}/support_faiss.index")
    support_df = pd.read_pickle(f"{MODELS_DIR}/support_df.pkl")

    groq_client = Groq(api_key=GROQ_API_KEY)

    return {
        "lang_clf": lang_clf,
        "lang_vectorizer": lang_vectorizer,
        "intent_clf": intent_clf,
        "intent_vectorizer": intent_vectorizer,
        "sentiment_pipe": sentiment_pipe,
        "embedder": embedder,
        "faiss_index": faiss_index,
        "support_df": support_df,
        "groq_client": groq_client,
    }


ID_TO_BUCKET = {0: "negative", 1: "neutral", 2: "positive"}


def retrieve(models, query, top_k=3):
    query_vec = models["embedder"].encode([query]).astype("float32")
    _, indices = models["faiss_index"].search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        results.append(
            {
                "instruction": models["support_df"].iloc[idx]["instruction"],
                "response": models["support_df"].iloc[idx]["response"],
            }
        )
    return results


def rag_answer(models, user_message, detected_sentiment="neutral"):
    retrieved = retrieve(models, user_message, top_k=3)
    context_text = "\n\n".join(f"Q: {r['instruction']}\nA: {r['response']}" for r in retrieved)

    system_prompt = f"""You are a helpful, professional customer support assistant for an online retailer.
Answer the customer's question using ONLY the information in the retrieved support responses below.
If the customer sounds frustrated ({detected_sentiment}), acknowledge that before answering.
If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing."""

    response = models["groq_client"].chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nCustomer question: \"{user_message}\""},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def route_message(models, user_message):
    lang_vec = models["lang_vectorizer"].transform([user_message])
    detected_lang = models["lang_clf"].predict(lang_vec)[0]

    sentiment_result = models["sentiment_pipe"](user_message)[0]
    sentiment_id = int(sentiment_result["label"].split("_")[-1])
    detected_sentiment = ID_TO_BUCKET[sentiment_id]

    if GREETING_PATTERNS.match(user_message.strip()):
        return {
            "language": str(detected_lang),
            "sentiment": str(detected_sentiment),
            "intent": "greeting",
            "response": "Hi there! How can I help you today?",
        }

    intent_vec = models["intent_vectorizer"].transform([user_message])
    detected_intent = models["intent_clf"].predict(intent_vec)[0]

    if detected_intent == "complaint":
        response_text = (
            "I'm really sorry to hear that. I'm escalating this to a human agent who will follow up with you shortly."
        )
    elif detected_intent not in RAG_INTENTS:
        response_text = "Thanks for reaching out! How can I help you today?"
    else:
        response_text = rag_answer(models, user_message, detected_sentiment)

    return {
        "language": str(detected_lang),
        "sentiment": str(detected_sentiment),
        "intent": str(detected_intent),
        "response": response_text,
    }


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="E-commerce Support Chatbot", page_icon="🛍️")
st.title("🛍️ Customer Support Chatbot")
st.caption("RAG-based support assistant — language detection, sentiment routing, intent classification, grounded Q&A")

if not GROQ_API_KEY:
    st.error("No GROQ_API_KEY found. Add it in Streamlit Cloud's app Secrets (or a local .env / env var).")
    st.stop()

models = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your order, refund, delivery, or account..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = route_message(models, prompt)
        st.markdown(result["response"])
        st.caption(f"intent: `{result['intent']}` · sentiment: `{result['sentiment']}` · language: `{result['language']}`")

    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
