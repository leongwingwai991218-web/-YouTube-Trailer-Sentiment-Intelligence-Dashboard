import streamlit as st
import joblib, torch, pandas as pd, yt_dlp, plotly.express as px, plotly.graph_objects as go
from transformers import AutoTokenizer
from wordcloud import WordCloud
import gdown, os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Disney Sentiment Intelligence Dashboard")

@st.cache_resource
def load_assets():
    model_path = 'model.pkl'
    # Downloading your model if not present
    if not os.path.exists(model_path):
        gdown.download('https://drive.google.com/uc?export=download&id=13Lb2WECIxXT5NpayZVRx2wXerp8O65fF', model_path, quiet=True)
    model = joblib.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()
    return model, tokenizer

@st.cache_data(ttl=600)
def analyze_trailer(url, max_comments):
    # UPGRADE: Added browser-like headers to bypass YouTube bot detection
    ydl_opts = {
        'quiet': True, 
        'getcomments': True, 
        'skip_download': True,
        'ignoreerrors': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        comments = sorted(info.get('comments', []), key=lambda x: x.get('timestamp', 0), reverse=True)
        texts = [c.get('text', '') for c in comments][:max_comments]

    if not texts: return None, None
    
    model, tokenizer = load_assets()
    
    # BATCH INFERENCE: Process in chunks to prevent lag
    batch_size = 25 
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            out = model(**inputs).logits
            preds.extend(torch.argmax(out, dim=1).numpy())
            confs.extend(torch.softmax(out, dim=1).detach().numpy().max(axis=1))
    
    df = pd.DataFrame({'text': texts, 'sentiment': [["Negative", "Neutral", "Positive"][p] for p in preds], 'conf': confs})
    return df, {'title': info.get('title'), 'thumb': info.get('thumbnail'), 'uploader': info.get('uploader'), 'views': info.get('view_count', 0)}

# --- MAIN UI ---
st.title("🏰 Disney Trailer Sentiment Intelligence Dashboard")
tab1, tab2 = st.tabs(["📊 Real Time Trailer Analysis", "🔍 Individual Comment Check"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    url = col_a.text_input("Enter Disney Trailer URL:")
    max_c = col_b.slider("Number of comments to analyze", 50, 200, 100)

    if st.button("Run Sentiment Analysis"):
        with st.spinner("Processing..."):
            df, meta = analyze_trailer(url, max_c)
            if df is None: st.error("No comments found. The video might have restricted access.")
            else:
                # Metrics and layout
                m1, m2, m3 = st.columns(3)
                m1.metric("Sentiment Index", f"{(df['sentiment'] == 'Positive').mean() * 100 - (df['sentiment'] == 'Negative').mean() * 100:.1f}")
                m2.metric("Positive Advocacy", f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%")
                m3.metric("Friction Index", f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%")
                
                # Faster Visualization
                st.subheader("Buzzword Analysis")
                clean_text = " ".join(df['text'])
                wc = WordCloud(width=800, height=200, background_color='white').generate(clean_text)
                st.image(wc.to_image(), use_container_width=True)
                
                st.subheader("Detailed Breakdown")
                st.dataframe(df[['text', 'sentiment', 'conf']], use_container_width=True)
with tab2:



    st.subheader("Individual Comment Check")
    # 1. Add a visual guide/scale

    st.markdown("""
    **Sentiment Scale Guide:**
    * **Negative 😡**: High levels of criticism, sarcasm, or dissatisfaction.
    * **Neutral 😐**: Factual statements, questions, or non-emotional engagement.
    * **Positive 😊**: Praise, excitement, or strong endorsement.
    """)


    txt = st.text_area(

        "Enter a comment to analyze:",

        placeholder="e.g., 'This movie looks amazing, can't wait to watch it!'"

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



                



                # Dynamic interpretation message



                if probs[p] > 0.8:



                    st.info("The model is highly confident in this classification.")



                elif probs[p] < 0.5:



                    st.warning("The model's confidence is low; this comment may be ambiguous.")



                    



            with col_chart:



                fig_bar = px.bar(x=["Negative", "Neutral", "Positive"], y=probs,



                                 color=["Negative", "Neutral", "Positive"],



                                 color_discrete_map={"Negative": "#D32F2F", "Neutral": "#FFC107",



                                                     "Positive": "#2E7D32"})



                fig_bar.update_layout(showlegend=False, height=300)



                st.plotly_chart(fig_bar, use_container_width=True)
