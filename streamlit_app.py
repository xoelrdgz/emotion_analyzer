import streamlit as st
import sys
import os
import json  # Añadir para json.dumps()
import pandas as pd  # Añadir para pd.DataFrame()
from emotion_analyzer import EmotionAnalyzer, Config
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Forzar el uso del backend no interactivo

# Configurar logging para suprimir advertencias específicas
import logging
logging.getLogger('torch._C').setLevel(logging.ERROR)
logging.getLogger('streamlit.watcher').setLevel(logging.ERROR)

# Inicializar el historial de análisis si no existe
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Inicializar variable para mantener el texto
if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = ""

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
    # Ejemplos predefinidos
    st.subheader("Try with an example")
    examples = {
        "Positive": "I'm really happy about the progress we've made. This project is turning out amazing!",
        "Negative": "I'm disappointed with the results. Everything went wrong and I feel frustrated.",
        "Mixed": "While I'm excited about the new opportunities, I'm also nervous about the challenges ahead."
    }
    
    # Usar botones para una interacción más directa
    example_cols = st.columns(len(examples))
    for i, (label, example_text) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(f"{label} Example"):
                st.session_state.text_input_value = example_text
                st.rerun()  # <- Cambiar aquí
    
    # Si quieres un botón para limpiar el texto
    if st.button("Clear text"):
        st.session_state.text_input_value = ""
        st.rerun()  # <- Cambiar aquí

    # Text input con el valor actualizado
    text_input = st.text_area("Enter text to analyze:", 
                             value=st.session_state.text_input_value, 
                             height=150,
                             key="text_analysis_input")
    
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
                # Guardar en el historial
                st.session_state.analysis_history.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "sentiment": results["sentiment"]["label"],
                    "top_emotion": results["emotions"][0]["label"] if results["emotions"] else "None"
                })
                
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
                
                # Añade después de mostrar los resultados principales

                # Análisis de palabras
                st.subheader("Text Analysis")
                
                # Estadísticas básicas del texto
                word_count = len(text.split())
                char_count = len(text)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Word Count", word_count)
                with col4:
                    st.metric("Character Count", char_count)
                
                # Nube de palabras
                if word_count > 5:
                    try:
                        from wordcloud import WordCloud, STOPWORDS
                        import numpy as np
                                                
                        # Usar stopwords en inglés para eliminar palabras comunes
                        stopwords = set(STOPWORDS)
                        
                        # Asegurarse de que tenemos texto para analizar
                        if text and len(text.strip()) > 0:
                            wordcloud = WordCloud(
                                width=800, 
                                height=400, 
                                background_color='white',
                                contour_width=1,
                                colormap='viridis',
                                max_words=100,
                                stopwords=stopwords,
                                min_font_size=10,
                                max_font_size=60
                            ).generate(text)
                            
                            # Mostrar la nube de palabras como imagen
                            # (eliminado el st.pyplot para evitar duplicados)
                            st.image(wordcloud.to_array(), caption='Word Cloud')
                        else:
                            st.warning("El texto está vacío, no se puede generar la nube de palabras.")
                    except Exception as e:
                        st.error(f"Error al generar la nube de palabras: {str(e)}")
                        st.info("Asegúrate de tener el paquete wordcloud instalado con: pip install wordcloud")
                else:
                    st.info("Se necesitan más de 5 palabras para generar una nube de palabras significativa.")
                
                # Generate visualization (no mostrada)
                st.subheader("Visualization Download")
                fig = analyzer.visualize_results(
                    sentiment["label"], 
                    sentiment["score"], 
                    results["emotions"],
                    show_plot=False  # No mostrar directamente para evitar el plt.show()
                )

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

                # Añadir después del botón de descarga de la visualización

                # Exportar resultados en JSON
                json_results = json.dumps(results, indent=2)
                st.download_button(
                    label="Download Results as JSON",
                    data=json_results,
                    file_name="emotion_analysis_results.json",
                    mime="application/json"
                )

    # Mostrar historial de análisis
    if st.session_state.analysis_history:
        st.subheader("Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df)
        
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()  # <- Cambiar aquí