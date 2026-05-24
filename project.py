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

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="YouTube Trailer Sentiment Intelligence Dashboard")


@st.cache_resource
def load_assets():
    model_path = 'model.pkl'
    # Check if model exists; if not, download it
    if not os.path.exists(model_path):
        # Use a public Google Drive URL
        url = 'https://365umedumy-my.sharepoint.com/:u:/g/personal/24074889_siswa365_um_edu_my/IQDAIlLiZQbRQL-3_-k-w4qbAeQLmDBhZOObUybFHYYIg74?e=fRsbi2'
        gdown.download(url, model_path, quiet=False)
    
    # Load model
    model = joblib.load(model_path)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()
    return model, tokenizer


@st.cache_data(ttl=3600)
def analyze_trailer(url, max_comments):
    ydl_opts = {'quiet': True, 'getcomments': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Robust handling for missing or disabled comments
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
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.inference_mode():
        outputs = model(**inputs).logits
        preds = torch.argmax(outputs, dim=1).numpy()
        probs = torch.softmax(outputs, dim=1).detach().numpy()
    df = pd.DataFrame(
        {'text': texts, 'sentiment': [["Negative", "Neutral", "Positive"][p] for p in preds], 'conf': probs.max(axis=1),
         'probs': list(probs)})
    return df, meta


# --- MAIN UI ---
st.title("🎬 YouTube Trailer Sentiment Intelligence Dashboard")
tab1, tab2 = st.tabs(["📊 Marketing Analytics", "🔍 Individual Audit"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    url = col_a.text_input("Enter Trailer URL:")
    max_c = col_b.slider("Number of comments", 50, 500, 100)

    if st.button("Run Comprehensive Audit", type="primary"):
        df, meta = analyze_trailer(url, max_c)

        if df is None:
            st.error("No comments found for this video. Please try a different URL.")
        else:
            # Metadata Header
            c_img, c_text = st.columns([1, 4])
            c_img.image(meta['thumb'], use_container_width=True)
            c_text.subheader(meta['title'])
            c_text.write(
                f"**Channel:** {meta['uploader']} | **Views:** {meta['views']:,} | **Likes:** {meta['likes']:,}")

            # Metrics
            metrics = {"Sentiment Index": (df['sentiment'] == 'Positive').mean() * 100 - (
                        df['sentiment'] == 'Negative').mean() * 100}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sentiment Index", f"{metrics['Sentiment Index']:.1f}")
            m2.metric("Advocacy Rate", f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%")
            m3.metric("Friction Index", f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%")
            m4.metric("Neutral Fatigue", f"{(df['sentiment'] == 'Neutral').mean() * 100:.1f}%")

            # 2:1 Layout
            left_col, right_col = st.columns([2, 1])

            with left_col:
                momentum_score = max(0, min(100, 50 + (metrics['Sentiment Index'] * 0.3)))

                # Dynamic color selection based on score
                if momentum_score < 33:
                    num_color = "#D32F2F"
                elif momentum_score < 66:
                    num_color = "#FFC107"
                else:
                    num_color = "#2E7D32"

                st.markdown("### <span style='color:black'>Recency Sentiment Momentum</span>", unsafe_allow_html=True)

                # Split gauge and legend
                g_col, l_col = st.columns([0.7, 0.3])
                with g_col:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(momentum_score, 1),
                        number={'font': {'color': num_color, 'size': 50}},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': num_color},
                               'steps': [{'range': [0, 33], 'color': "#F8D7DA"},
                                         {'range': [33, 66], 'color': "#FFF3CD"},
                                         {'range': [66, 100], 'color': "#D4EDDA"}]}
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with l_col:
                    st.markdown("""
                    <div style="margin-top: 50px; font-size: 13px;">
                    <b>Momentum Guide:</b><br>
                    🔴 Critical friction (0-33)<br>
                    🟡 Engagement tepid (34-66)<br>
                    🟢 High advocacy (67-100)
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("Buzzword Cloud")
                wc = WordCloud(width=800, height=150, background_color='white').generate(" ".join(df['text']))
                st.image(wc.to_array(), use_container_width=True)

            with right_col:
                st.subheader("Sentiment Distribution")
                fig_hist = px.histogram(df, x="sentiment", color="sentiment",
                                        color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107",
                                                            "Positive": "#2E7D32"})
                fig_hist.update_layout(height=650)
                st.plotly_chart(fig_hist, use_container_width=True)

            st.divider()
            st.subheader("📋 Detailed Comment Audit")
            st.dataframe(df[['text', 'sentiment', 'conf']], use_container_width=True)

with tab2:
    st.subheader("Individual Comment Audit")
    txt = st.text_area(
        "Enter a comment to analyze:",
        placeholder="e.g., 'This movie looks amazing, can't wait to watch it with my friends!'"
    )
    if st.button("Analyze"):
        if not txt.strip():
            st.warning("Please enter a comment to analyze.")
        else:
            model, tokenizer = load_assets()
            out = model(**tokenizer([txt], return_tensors="pt")).logits
            probs = torch.softmax(out, dim=1).detach().numpy()[0]
            p = torch.argmax(out, dim=1).item()

            col_res, col_chart = st.columns(2)
            with col_res:
                labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
                st.markdown(f"### Result: {labels[p]}")
                st.metric("Confidence Score", f"{probs[p] * 100:.2f}%")
            with col_chart:
                fig_bar = px.bar(x=["Negative", "Neutral", "Positive"], y=probs,
                                 color=["Negative", "Neutral", "Positive"],
                                 color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107",
                                                     "Positive": "#2E7D32"})
                st.plotly_chart(fig_bar, use_container_width=True)
