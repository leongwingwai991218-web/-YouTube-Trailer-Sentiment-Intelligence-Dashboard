import streamlit as st
import joblib
import torch
import pandas as pd
import yt_dlp
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer
from wordcloud import WordCloud
import gdown
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="YouTube Trailer Sentiment Intelligence")

# --- STABLE ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads model once and keeps it in memory using cache_resource."""
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        # Ensure this link is public: Anyone with link -> Viewer
        url = 'https://drive.google.com/uc?id=13Lb2WECIxXT5NpayZVRx2wXerp8O65fF'
        gdown.download(url, model_path, quiet=False)
    
    model = joblib.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()
    return model, tokenizer

# --- ANALYSIS LOGIC ---
@st.cache_data(ttl=600)
def analyze_trailer(url, max_comments):
    ydl_opts = {
        'quiet': True, 
        'getcomments': True, 
        'skip_download': True,
        'ignoreerrors': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # --- ROBUSTNESS GUARD ---
        if not isinstance(info, dict) or 'comments' not in info:
            return None, None
            
        comments_data = info.get('comments', [])
        
        sorted_comments = sorted(comments_data, key=lambda x: x.get('timestamp', 0), reverse=True)
        texts = [c.get('text', '') for c in sorted_comments][:max_comments]

        if not texts: return None, None
        
        meta = {
            'title': info.get('title', 'Unknown'),
            'thumb': info.get('thumbnail'),
            'uploader': info.get('uploader'),
            'views': info.get('view_count', 0)
        }

    model, tokenizer = load_assets()
    
    # Batch processing
    batch_size = 32
    preds, probs_max = [], []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs).logits
            preds.extend(torch.argmax(outputs, dim=1).numpy())
            probs_max.extend(torch.softmax(outputs, dim=1).max(dim=1).values.numpy())
        
    df = pd.DataFrame({'text': texts, 'sentiment': [["Negative", "Neutral", "Positive"][p] for p in preds], 'conf': probs_max})
    return df, meta

# --- MAIN UI ---
st.title("🎬 YouTube Trailer Sentiment Intelligence Dashboard")
tab1, tab2 = st.tabs(["📊 Real Time Trailer Analysis", "🔍 Individual Comment Check"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    url = col_a.text_input("Enter Trailer URL:")
    max_c = col_b.slider("Number of comments to analyze", 50, 500, 100)

    if st.button("Run Sentiment Analysis", type="primary"):
        with st.spinner("Analyzing..."):
            df, meta = analyze_trailer(url, max_c)
            if df is None: st.error("No comments found or invalid URL.")
            else:
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

                # MOMENTUM GAUGE
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

                # Charts
                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.subheader("Buzzword Cloud")
                    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['text']))
                    st.image(wc.to_array(), use_container_width=True)
                with row2_col2:
                    st.subheader("Sentiment Distribution")
                    st.plotly_chart(px.histogram(df, x="sentiment", color="sentiment", height=400, color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107", "Positive": "#2E7D32"}), use_container_width=True)

                st.subheader("📋 Detailed Comment Analysis (Newest First)")
                st.dataframe(df[['text', 'sentiment', 'conf']], use_container_width=True)

with tab2:
    st.subheader("Individual Comment Check")
    txt = st.text_area("Enter a comment to analyze:")
    if st.button("Analyze"):
        if not txt.strip(): st.warning("Please enter a comment.")
        else:
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
                fig_bar.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
