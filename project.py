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
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?export=download&id=13Lb2WECIxXT5NpayZVRx2wXerp8O65fF'
        gdown.download(url, model_path, quiet=False)
    
    model = joblib.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()
    return model, tokenizer

# --- ANALYSIS LOGIC (Optimized) ---
@st.cache_data(ttl=3600)
def analyze_trailer(url, max_comments):
    ydl_opts = {'quiet': True, 'getcomments': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        comments_data = info.get('comments') or []
        texts = [c.get('text', '') for c in comments_data][:max_comments]

        if not texts: return None, None

        meta = {
            'title': info.get('title', 'Unknown'),
            'thumb': info.get('thumbnail'),
            'uploader': info.get('uploader'),
            'views': info.get('view_count', 0),
            'likes': info.get('like_count', 0)
        }

    model, tokenizer = load_assets()
    if model is None: return None, meta
        
    batch_size = 50
    preds, probs_max = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            outputs = model(**inputs).logits
            preds.extend(torch.argmax(outputs, dim=1).numpy())
            probs_max.extend(torch.softmax(outputs, dim=1).detach().numpy().max(axis=1))
        
    df = pd.DataFrame({'text': texts, 'sentiment': [["Negative", "Neutral", "Positive"][p] for p in preds], 'conf': probs_max})
    return df, meta

# --- MAIN UI ---
st.title("🎬 YouTube Trailer Sentiment Intelligence Dashboard")
tab1, tab2 = st.tabs(["📊 Marketing Analytics", "🔍 Individual Audit"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    url = col_a.text_input("Enter Trailer URL:")
    max_c = col_b.slider("Number of comments", 50, 500, 100)

    if st.button("Run Comprehensive Audit", type="primary"):
        with st.spinner("Analyzing..."):
            df, meta = analyze_trailer(url, max_c)
            if df is None:
                st.error("No comments found or model failed to load.")
            else:
                # Metadata
                c_img, c_text = st.columns([1, 4])
                c_img.image(meta['thumb'], use_container_width=True)
                c_text.subheader(meta['title'])
                c_text.write(f"**Channel:** {meta['uploader']} | **Views:** {meta['views']:,}")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sentiment Index", f"{(df['sentiment'] == 'Positive').mean() * 100 - (df['sentiment'] == 'Negative').mean() * 100:.1f}")
                m2.metric("Advocacy Rate", f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%")
                m3.metric("Friction Index", f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%")
                m4.metric("Neutral Fatigue", f"{(df['sentiment'] == 'Neutral').mean() * 100:.1f}%")

                # ROW 1: Momentum + Guide
                st.subheader("Recency Sentiment Momentum")
                g1, g2 = st.columns([2, 1])
                momentum_score = max(0, min(100, 50 + (((df['sentiment'] == 'Positive').mean() * 100 - (df['sentiment'] == 'Negative').mean() * 100) * 0.3)))
                num_color = "#D32F2F" if momentum_score < 33 else "#FFC107" if momentum_score < 66 else "#2E7D32"
                
                with g1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=round(momentum_score, 1),
                        number={'font': {'color': num_color, 'size': 50}},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': num_color},
                               'steps': [{'range': [0, 33], 'color': "#F8D7DA"}, {'range': [33, 66], 'color': "#FFF3CD"}, {'range': [66, 100], 'color': "#D4EDDA"}]}
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with g2:
                    st.markdown("""<div style="margin-top: 50px;"><b>Momentum Guide:</b><br>🔴 Critical friction (0-33)<br>🟡 Engagement tepid (34-66)<br>🟢 High advocacy (67-100)</div>""", unsafe_allow_html=True)

                # ROW 2: Word Cloud + Distribution
                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.subheader("Buzzword Cloud")
                    clean_text = "".join([char for char in " ".join(df['text']) if ord(char) < 128])
                    wc = WordCloud(width=800, height=400, background_color='white').generate(clean_text)
                    st.image(wc.to_array(), use_container_width=True)
                with row2_col2:
                    st.subheader("Sentiment Distribution")
                    fig_hist = px.histogram(df, x="sentiment", color="sentiment", height=400,
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
        col_res, col_chart = st.columns(2)
        with col_res:
            st.markdown(f"### Result: {['Negative 😡', 'Neutral 😐', 'Positive 😊'][p]}")
            st.metric("Confidence Score", f"{probs[p] * 100:.2f}%")
        with col_chart:
            fig_bar = px.bar(x=["Negative", "Neutral", "Positive"], y=probs, color=["Negative", "Neutral", "Positive"],
                             color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107", "Positive": "#2E7D32"})
            st.plotly_chart(fig_bar, use_container_width=True)
