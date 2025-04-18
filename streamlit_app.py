# frontend/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from emotion_analyzer import EmotionAnalyzer, Config
import logging

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Emotion Analyzer | AI-powered Text Analysis",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS styles from file."""
    css_file = Path(__file__).parent / "static" / "style.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Emoji and color mappings for emotions and sentiments
EMOJI_MAP = {
    "joy": "😊", "love": "❤️", "anger": "😠", "fear": "😨",
    "sadness": "😢", "surprise": "😲", "neutral": "😐",
    "worry": "😟", "happy": "😄", "hate": "😡", "admiration": "🤩",
    "approval": "👍", "caring": "🤗", "confusion": "🤔",
    "curiosity": "🧐", "desire": "🥰", "disappointment": "😞",
    "disapproval": "👎", "disgust": "🤢", "embarrassment": "😳",
    "excitement": "🤪", "gratitude": "🙏", "grief": "😭",
    "nervousness": "😰", "optimism": "🌟", "pride": "🦁",
    "realization": "💡", "relief": "😌", "remorse": "😔",
    "annoyance": "😤", "awe": "🥺", "anticipation": "🤗",
    "distraction": "🤪", "empathy": "💕", "enthusiasm": "✨",
    "indifference": "🤷",
}

COLORS = {
    "positive": "#2ecc71", "negative": "#e74c3c", 
    "neutral": "#f1c40f", "neutral_sentiment": "#f1c40f",
    "primary": "#3f51b5", "secondary": "#f50057",
    
    "joy": "#2ecc71", "love": "#e84393", "anger": "#e74c3c",
    "fear": "#8e44ad", "sadness": "#3498db", "surprise": "#f1c40f",
    "worry": "#e67e22", "happy": "#2ecc71", "hate": "#c0392b",
    "admiration": "#27ae60", "approval": "#16a085", "caring": "#9b59b6",
    "confusion": "#34495e", "curiosity": "#3498db", "desire": "#e74c3c",
    "disappointment": "#95a5a6", "disapproval": "#c0392b",
    "disgust": "#d35400", "embarrassment": "#e67e22",
    "excitement": "#f1c40f", "gratitude": "#27ae60", "grief": "#2c3e50",
    "nervousness": "#9b59b6", "optimism": "#2ecc71", "pride": "#f39c12",
    "realization": "#3498db", "relief": "#1abc9c", "remorse": "#95a5a6",
    "annoyance": "#e74c3c", "neutral": "#95a5a6"
}

@st.cache_resource
def load_emotion_analyzer():
    """Initialize and cache the EmotionAnalyzer instance."""
    logger.info("Attempting to load EmotionAnalyzer...")
    config = Config()
    config.show_vis = False
    analyzer = EmotionAnalyzer(config)
    if analyzer.initialize_models():
        logger.info("EmotionAnalyzer loaded successfully.")
        return analyzer
    logger.error("Failed to initialize EmotionAnalyzer.")
    st.error("Fatal Error: Could not load analysis models. Please check logs.")
    return None

analyzer = load_emotion_analyzer()

if analyzer is None:
    st.stop()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'show_examples' not in st.session_state:
    st.session_state.show_examples = False

if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = ""

EXAMPLES = [
    "I'm having a wonderful day at the beach. The sun is shining and the waves are perfect!",
    "I'm really disappointed with the service we received. The staff was rude and unhelpful.",
    "I'm not sure how I feel about this situation. It's complicated and I need more time to think.",
    "It terrifies me to think about what might happen tomorrow. I can't sleep from the anxiety.",
    "I'm so proud of my daughter for winning the competition. She worked so hard for this!"
]

def get_emotion_label_with_emoji(emotion_label: str) -> str:
    """Get formatted emotion label with emoji"""
    emoji = EMOJI_MAP.get(emotion_label.lower(), "")
    return f"{emoji} {emotion_label}"

def create_emotions_bar_chart(emotions):
    """Creates a bar chart for detected emotions"""
    if not emotions:
        return None
    
    df = pd.DataFrame(emotions)
    df = df.sort_values('score', ascending=True)
    df['label'] = df['label'].apply(get_emotion_label_with_emoji)
    colors = [COLORS.get(emotion.lower(), COLORS['neutral']) for emotion in df['label']]
    
    fig = px.bar(
        df, x='score', y='label',
        orientation='h',
        text='score',
        labels={'score': 'Confidence (%)', 'label': 'Emotion'},
        height=350,
    )
    
    fig.update_traces(
        marker_color=colors,
        texttemplate='%{text:.1f}%',
        textposition='outside',
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, max(df['score'])*1.15]),
        font=dict(size=12)
    )
    
    return fig

def create_emotion_radar_chart(emotions):
    """Creates a radar chart for detected emotions"""
    if not emotions or len(emotions) < 2:
        return None
    
    labels = [f"{get_emotion_label_with_emoji(emo['label'])} ({emo['score']:.1f}%)" for emo in emotions]
    values = [emo['score'] for emo in emotions]
    labels.append(labels[0])
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color=COLORS['primary'], width=2),
        fillcolor='rgba(63, 81, 181, 0.5)',
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.1])),
        showlegend=False,
        margin=dict(l=70, r=70, t=20, b=20),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig

def format_timestamp():
    """Returns a formatted timestamp"""
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

def use_example(example_idx):
    """Sets a predefined example in the text area"""
    st.session_state.text_input_value = EXAMPLES[example_idx]

def clear_history():
    """Clears analysis history"""
    st.session_state.analysis_history = []

def display_sentiment_card(sentiment, col):
    """Display sentiment analysis results in a card"""
    with col:
        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]
        sentiment_category = sentiment.get("category", "neutral")
        
        st.markdown(f"""
        <div class="sentiment-card {sentiment_category}">
            <h1>{sentiment_label}</h1>
            <p>Category: <strong>{sentiment_category.upper()}</strong></p>
            <p>Confidence: <strong>{sentiment_score:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

def display_header():
    """Display page header"""
    st.markdown('<p class="header-title">💭 Emotion Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Analyze emotions and sentiments in any text</p>', unsafe_allow_html=True)

# Main header
display_header()

# Text input section
st.subheader("📝 Enter your text")

# Examples panel
if st.button("View examples" if not st.session_state.show_examples else "Hide examples"):
    st.session_state.show_examples = not st.session_state.show_examples

if st.session_state.show_examples:
    example_cols = st.columns(len(EXAMPLES))
    for i, (col, example) in enumerate(zip(example_cols, EXAMPLES)):
        with col:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                use_example(i)

# Input options
# Use the 'key' parameter to link the text_area to session_state
st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Write or paste your text here to analyze its emotions and sentiment...",
    key="text_input_value" # Link to st.session_state.text_input_value
)

# Add threshold slider
threshold = st.slider(
    "Emotion confidence threshold (%)",
    min_value=1.0,
    max_value=100.0,
    value=3.0,
    step=0.5,
    help="Only show emotions with confidence scores above this threshold"
)

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])

with col2:
    analyze_button = st.button("🔍 Analyze", use_container_width=True)

def process_analysis(text):
    """Process text analysis and update UI"""
    global analyzer
    if analyzer is None:
         st.error("Analysis models are not available.")
         return

    with st.spinner("Analyzing text..."):
        analyzer.config.emotion_threshold = threshold
        results = analyzer.analyze(text)

        if results:
            results["emotions"] = [
                emotion for emotion in results["emotions"]
                if emotion["score"] >= threshold
            ]

            results["timestamp"] = format_timestamp()
            results["text"] = text
            st.session_state.analysis_history.insert(0, results)

            st.success("Analysis completed!")

            st.subheader("📊 Analysis Results")

            col_sentiment, col_emotions = st.columns([1, 2])

            display_sentiment_card(results["sentiment"], col_sentiment)

            with col_emotions:
                if results["emotions"]:
                    emotions_fig = create_emotions_bar_chart(results["emotions"])
                    st.plotly_chart(emotions_fig, use_container_width=True)
                    st.info(f"Detected {len(results['emotions'])} emotions above {threshold}% confidence threshold")
                else:
                    st.info(f"No emotions detected above {threshold}% confidence threshold")

            if results["emotions"] and len(results["emotions"]) >= 2:
                st.subheader("🔍 Advanced Visualization")
                radar_fig = create_emotion_radar_chart(results["emotions"])
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.error("Analysis failed. Please try again.")

if analyze_button and (st.session_state.text_input_value or uploaded_file):
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.session_state.text_input_value
        
    if text.strip():
        process_analysis(text.strip())
    else:
        st.error("Please enter some text to analyze")

if st.session_state.analysis_history:
    st.divider()
    st.subheader("📜 Analysis History")

    for idx, analysis in enumerate(st.session_state.analysis_history[:5]):
        with st.expander(f"Analysis {len(st.session_state.analysis_history) - idx} - {analysis.get('timestamp', 'N/A')}"):
            st.markdown("**Analyzed text:**")
            st.markdown(f"> _{analysis.get('text', 'N/A')}_")

            col1, col2 = st.columns(2)

            with col1:
                sentiment = analysis["sentiment"]
                emotions = analysis["emotions"]

                st.markdown(f"**Sentiment:** {sentiment['label'].upper()} ({sentiment['score']:.2f}%)")

                st.markdown("**Detected emotions:**")
                if emotions:
                    for emotion in emotions:
                        emotion_color = COLORS.get(emotion['label'].lower(), COLORS['neutral'])
                        st.markdown(
                            f"""<div class="emotion-item">
                                <span class="emotion-label">{get_emotion_label_with_emoji(emotion['label'])}</span>
                                <div class="emotion-bar" style="width: {min(emotion['score'], 100)}%; background-color: {emotion_color};"></div>
                                <span class="emotion-value">{emotion['score']:.1f}%</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No emotions above threshold.")

            with col2:
                if emotions:
                    emotions_fig = create_emotions_bar_chart(emotions)
                    if emotions_fig:
                        st.plotly_chart(emotions_fig, use_container_width=True, key=f"history_{idx}_chart")
                else:
                    st.write("No chart to display.")

    if len(st.session_state.analysis_history) > 0:
        if st.button("Clear History"):
            clear_history()
            st.rerun()