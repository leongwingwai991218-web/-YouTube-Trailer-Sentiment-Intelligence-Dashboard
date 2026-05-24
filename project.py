import streamlit as st
import joblib
import torch
import pandas as pd
import yt_dlp
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gdown
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="YouTube Trailer Sentiment Intelligence Dashboard")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model_path = 'model.pkl'
    # Download model if not present using a valid direct download Google Drive URL
    if not os.path.exists(model_path):
        # IMPORTANT: Replace the ID below with your actual Google Drive File ID
        # The format MUST be: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
        url = 'https://drive.google.com/uc?export=download&id=13Lb2WECIxXT5NpayZVRx2wXerp8O65fF'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None, None
    
    try:
        model = joblib.load(model_path)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- ANALYSIS LOGIC ---
@st.cache_data(ttl=3600)
def analyze_trailer(url, max_comments):
    ydl_opts = {'quiet': True, 'getcomments': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        comments_data = info.get('comments') or []
        texts = [c.get('text', '') for c in comments_data][:max_comments]

        if not texts:
            return None, None

        meta = {
            'title': info.get('title', 'Unknown'),
            'thumb': info.get('thumbnail'),
            'uploader': info.get('uploader'),
            'views': info.get('view_count', 0),
            'likes': info.get('like_count', 0)
        }

    model, tokenizer = load_assets()
    if model is None: return None, meta
        
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.inference_mode():
        outputs = model(**inputs).logits
        preds = torch.argmax(outputs, dim=1).numpy()
        probs = torch.softmax(outputs, dim=1).detach().numpy()
        
    df = pd.DataFrame({
        'text': texts, 
        'sentiment': [["Negative", "Neutral", "Positive"][p] for p in preds], 
        'conf': probs.max(axis=1)
    })
    return df, meta

# --- MAIN UI ---
st.title("🎬 YouTube Trailer Sentiment Intelligence Dashboard")
tab1, tab2 = st.tabs(["📊 Marketing Analytics", "🔍 Individual Audit"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    url = col_a.text_input("Enter Trailer URL:")
    max_c = col_b.slider("Number of comments", 50, 500, 100)

    if st.button("Run Comprehensive Audit", type="primary"):
        with st.spinner("Analyzing comments..."):
            df, meta = analyze_trailer(url, max_c)
            
            if df is None:
                st.error("No comments found for this video or model failed to load.")
            else:
                # Metadata Header
                c_img, c_text = st.columns([1, 4])
                c_img.image(meta['thumb'], use_container_width=True)
                c_text.subheader(meta['title'])
                c_text.write(f"**Channel:** {meta['uploader']} | **Views:** {meta['views']:,}")

                # Metrics
                metrics = {"Sentiment Index": (df['sentiment'] == 'Positive').mean() * 100 - (df['sentiment'] == 'Negative').mean() * 100}
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sentiment Index", f"{metrics['Sentiment Index']:.1f}")
                m2.metric("Advocacy Rate", f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%")
                m3.metric("Friction Index", f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%")

                left_col, right_col = st.columns([2, 1])
                with left_col:
                    st.subheader("Buzzword Cloud")
                    # Clean text to remove non-ASCII characters to fix rectangle issue
                    clean_text = "".join([char for char in " ".join(df['text']) if ord(char) < 128])
                    wc = WordCloud(width=800, height=150, background_color='white').generate(clean_text)
                    st.image(wc.to_array(), use_container_width=True)

                with right_col:
                    st.subheader("Sentiment Distribution")
                    fig_hist = px.histogram(df, x="sentiment", color="sentiment",
                                            color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107", "Positive": "#2E7D32"})
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.subheader("📋 Detailed Comment Audit")
                st.dataframe(df[['text', 'sentiment', 'conf']], use_container_width=True)

with tab2:
    st.subheader("Individual Comment Audit")
    txt = st.text_area("Enter a comment to analyze:")
    if st.button("Analyze"):
        model, tokenizer = load_assets()
        out = model(**tokenizer([txt], return_tensors="pt")).logits
        probs = torch.softmax(out, dim=1).detach().numpy()[0]
        p = torch.argmax(out, dim=1).item()
        st.write(f"Result: {['Negative', 'Neutral', 'Positive'][p]}")
