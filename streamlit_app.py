import streamlit as st
import sys
import os
import json
import pandas as pd
from emotion_analyzer import EmotionAnalyzer, Config
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')

import logging
logging.getLogger('torch._C').setLevel(logging.ERROR)
logging.getLogger('streamlit.watcher').setLevel(logging.ERROR)

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = ""

st.set_page_config(page_title="Emotion Analyzer", layout="wide")

threshold = st.sidebar.slider(
    "Emotion confidence threshold (%)", 0, 100, 3,
    help="Filter out emotions below this confidence level"
)
wc_colormap = st.sidebar.selectbox(
    "Word Cloud colormap",
    ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    index=0
)

@st.cache_resource
def get_analyzer():
    config = Config()
    analyzer = EmotionAnalyzer(config)
    if analyzer.initialize_models():
        return analyzer
    else:
        st.error("Failed to initialize models")
        return None

st.title("💭 Emotion & Sentiment Analyzer")
st.write("Analyze emotions and sentiment in text using BERT models")

analyzer = get_analyzer()

if analyzer:
    st.subheader("Try with an example")
    examples = {
        "Positive": "I'm really happy about the progress we've made. This project is turning out amazing!",
        "Negative": "I'm disappointed with the results. Everything went wrong and I feel frustrated.",
        "Mixed": "While I'm excited about the new opportunities, I'm also nervous about the challenges ahead."
    }
    
    example_cols = st.columns(len(examples))
    for i, (label, example_text) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(f"{label} Example"):
                st.session_state.text_input_value = example_text
                st.rerun()
    
    if st.button("Clear text"):
        st.session_state.text_input_value = ""
        st.rerun()

    text_input = st.text_area("Enter text to analyze:", 
                             value=st.session_state.text_input_value, 
                             height=150,
                             key="text_analysis_input")
    
    uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"])
    
    analyze_button = st.button("Analyze")
    
    if analyze_button and (text_input or uploaded_file):
        with st.spinner("Analyzing..."):
            if uploaded_file:
                text = uploaded_file.read().decode("utf-8")
            else:
                text = text_input
                
            results = analyzer.analyze(text)
            
            if results:
                st.session_state.analysis_history.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "sentiment": results["sentiment"]["label"],
                    "top_emotion": results["emotions"][0]["label"] if results["emotions"] else "None"
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Analysis")
                    sentiment = results["sentiment"]
                    st.metric("Sentiment", sentiment["label"].upper(), f"{sentiment['score']:.2f}%")
                    
                    category = analyzer.get_sentiment_category(sentiment["label"])
                    st.info(f"Category: {category.upper()}")
                
                with col2:
                    st.subheader("Emotion Analysis")
                    
                    filtered_emotions = [
                        emo for emo in results["emotions"]
                        if emo["score"] * 100 >= threshold
                    ]
                    if filtered_emotions:
                        for emotion in filtered_emotions:
                            score = emotion["score"] * 100
                            st.progress(score / 100)
                            st.write(f"{emotion['label']}: {score:.2f}%")
                    else:
                        st.info("No emotions exceed the confidence threshold.")
                
                st.subheader("Text Analysis")
                
                word_count = len(text.split())
                char_count = len(text)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Word Count", word_count)
                with col4:
                    st.metric("Character Count", char_count)
                
                if word_count > 5:
                    try:
                        from wordcloud import WordCloud, STOPWORDS
                        import numpy as np
                                                
                        stopwords = set(STOPWORDS)
                        
                        if text and len(text.strip()) > 0:
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                contour_width=1,
                                colormap=wc_colormap,
                                max_words=100,
                                stopwords=stopwords,
                                min_font_size=10,
                                max_font_size=60
                            ).generate(text)
                            
                            st.image(wordcloud.to_array(), caption='Word Cloud')
                        else:
                            st.warning("Text is empty or invalid for word cloud generation.")
                    except Exception as e:
                        st.error(f"Error generating word cloud: {e}")
                        st.info("Please ensure you have the wordcloud library installed.")
                else:
                    st.info("Not enough words to generate a word cloud.")
                
                st.subheader("Visualization Download")
                fig = analyzer.visualize_results(
                    sentiment["label"], 
                    sentiment["score"], 
                    results["emotions"],
                    show_plot=False
                )

                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)

                st.download_button(
                    label="Download Visualization",
                    data=buf,
                    file_name="emotion_analysis.png",
                    mime="image/png"
                )

                json_results = json.dumps(results, indent=2)
                st.download_button(
                    label="Download Results as JSON",
                    data=json_results,
                    file_name="emotion_analysis_results.json",
                    mime="application/json"
                )

    if st.session_state.analysis_history:
        st.subheader("Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df)
        
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()