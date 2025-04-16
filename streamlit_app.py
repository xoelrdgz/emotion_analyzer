import streamlit as st
import sys
import os
from emotion_analyzer import EmotionAnalyzer, Config
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Forzar el uso del backend no interactivo

st.set_page_config(page_title="Emotion Analyzer", layout="wide")

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
    # Text input
    text_input = st.text_area("Enter text to analyze:", height=150)
    
    # File upload
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
                # Display results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Analysis")
                    sentiment = results["sentiment"]
                    st.metric("Sentiment", sentiment["label"].upper(), f"{sentiment['score']:.2f}%")
                    
                    category = analyzer.get_sentiment_category(sentiment["label"])
                    st.info(f"Category: {category.upper()}")
                
                with col2:
                    st.subheader("Emotion Analysis")
                    
                    for emotion in results["emotions"]:
                        score = emotion["score"] * 100
                        st.progress(score / 100)
                        st.write(f"{emotion['label']}: {score:.2f}%")
                
                # Generate visualization
                st.subheader("Visualization")
                fig = analyzer.visualize_results(
                    sentiment["label"], 
                    sentiment["score"], 
                    results["emotions"],
                    show_plot=False  # No mostrar directamente para evitar el plt.show()
                )

                # Mostrar la figura en Streamlit
                st.pyplot(fig)

                # Save visualization to buffer
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)

                # Download button for visualization
                st.download_button(
                    label="Download Visualization",
                    data=buf,
                    file_name="emotion_analysis.png",
                    mime="image/png"
                )