# frontend/streamlit_app.py
import streamlit as st
import requests
import time
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# Matplotlib configuration
matplotlib.use('Agg')

# Page configuration
st.set_page_config(
    page_title="Emotion Analyzer | AI-powered Text Analysis",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and configuration
DEFAULT_API_URL = "http://localhost:8000"
EMOJI_MAP = {
    "joy": "😊", "love": "❤️", "anger": "😠", "fear": "😨",
    "sadness": "😢", "surprise": "😲", "worry": "😟",
    "neutral": "😐", "happy": "😄", "hate": "😡"
}

COLORS = {
    "joy": "#2ecc71", "love": "#e84393", "anger": "#e74c3c",
    "fear": "#8e44ad", "sadness": "#3498db", "surprise": "#f1c40f",
    "neutral": "#95a5a6", "worry": "#e67e22", "happy": "#2ecc71",
    "hate": "#c0392b", "positive": "#2ecc71", "negative": "#e74c3c",
    "neutral_sentiment": "#f1c40f", "primary": "#3f51b5", "secondary": "#f50057"
}

# Initialize session state variables
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = ""

if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if 'api_url' not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "unknown"

if 'show_examples' not in st.session_state:
    st.session_state.show_examples = False

# Examples for quick analysis
EXAMPLES = [
    "I'm having a wonderful day at the beach. The sun is shining and the waves are perfect!",
    "I'm really disappointed with the service we received. The staff was rude and unhelpful.",
    "I'm not sure how I feel about this situation. It's complicated and I need more time to think.",
    "It terrifies me to think about what might happen tomorrow. I can't sleep from the anxiety.",
    "I'm so proud of my daughter for winning the competition. She worked so hard for this!"
]

# Utility functions
def check_api_connection():
    """Checks the connection with the backend API"""
    try:
        response = requests.get(f"{st.session_state.api_url}/")
        if response.status_code == 200:
            st.session_state.connection_status = "connected"
            return True
        else:
            st.session_state.connection_status = "error"
            return False
    except:
        st.session_state.connection_status = "error"
        return False

def submit_analysis(text, threshold):
    """Sends text for analysis to the backend"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/analyze",
            json={"text": text, "threshold": threshold / 100}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.job_id = data["job_id"]
            return True
        else:
            st.error(f"Analysis submission error: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return False

def get_analysis_result():
    """Gets analysis results from the backend"""
    if not st.session_state.job_id:
        return None
    
    try:
        response = requests.get(f"{st.session_state.api_url}/result/{st.session_state.job_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "pending", "error_code": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def create_emotions_bar_chart(emotions):
    """Creates a bar chart for detected emotions"""
    if not emotions:
        return None
    
    # Create dataframe for the chart
    df = pd.DataFrame(emotions)
    df = df.sort_values('score', ascending=True)  # Sort from lowest to highest
    
    # Assign colors to each emotion
    colors = [COLORS.get(emotion.lower(), COLORS['neutral']) for emotion in df['label']]
    
    # Create chart with plotly
    fig = px.bar(
        df, 
        x='score', 
        y='label',
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
    
    # Prepare data for radar chart
    labels = [f"{emo['label']} ({emo['score']:.1f}%)" for emo in emotions]
    values = [emo['score'] for emo in emotions]
    
    # Add first value at the end to close the polygon
    labels.append(labels[0])
    values.append(values[0])
    
    # Create chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color=COLORS['primary'], width=2),
        fillcolor='rgba(63, 81, 181, 0.5)',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
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

def export_to_json(data):
    """Exports analysis data to JSON format"""
    return json.dumps(data, indent=2)

def get_download_link(data, filename, text):
    """Generates a download link for the data"""
    json_str = export_to_json(data)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">📄 {text}</a>'
    return href

def use_example(example_idx):
    """Sets a predefined example in the text area"""
    st.session_state.text_input_value = EXAMPLES[example_idx]

def clear_history():
    """Clears analysis history"""
    st.session_state.analysis_history = []

# Function to change API URL
def set_api_url():
    st.session_state.api_url = st.session_state.api_url_input
    check_api_connection()

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    .sentiment-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .emotion-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-weight: 500;
        margin-right: 1rem;
        min-width: 100px;
    }
    .emotion-bar {
        height: 10px;
        border-radius: 5px;
        margin-right: 10px;
    }
    .emotion-value {
        font-weight: 500;
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: 500;
    }
    .example-btn {
        margin: 0.2rem;
        padding: 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
        cursor: pointer;
    }
    .example-btn:hover {
        background-color: #e9ecef;
    }
    .connection-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .connection-indicator.connected {
        background-color: #28a745;
    }
    .connection-indicator.error {
        background-color: #dc3545;
    }
    .connection-indicator.unknown {
        background-color: #ffc107;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0;
        color: #6c757d;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Connection Settings
    st.subheader("API Connection")
    col_status, col_indicator = st.columns([4, 1])
    
    with col_status:
        if st.session_state.connection_status == "connected":
            status_text = "Connected"
            status_class = "connected"
        elif st.session_state.connection_status == "error":
            status_text = "Disconnected"
            status_class = "error"
        else:
            status_text = "Unknown"
            status_class = "unknown"
        
        st.markdown(f"""
        <div>Status: <span class="connection-indicator {status_class}"></span> {status_text}</div>
        """, unsafe_allow_html=True)
    
    with col_indicator:
        if st.button("🔄"):
            check_api_connection()
    
    st.text_input("API URL:", value=st.session_state.api_url, key="api_url_input")
    st.button("Apply URL", on_click=set_api_url)
    
    # Analysis Settings
    st.divider()
    st.subheader("Analysis")
    threshold = st.slider(
        "Confidence threshold for emotions (%)", 0, 100, 3,
        help="Filter emotions with confidence below this threshold"
    )
    
    # History Management
    st.divider()
    st.subheader("History")
    history_count = len(st.session_state.analysis_history)
    st.write(f"Saved analyses: {history_count}")
    st.button("🗑️ Clear history", on_click=clear_history, disabled=history_count == 0)
    
    # About section
    st.divider()
    st.subheader("About")
    st.markdown("""
    **Emotion Analyzer** is an AI-powered tool that analyzes emotions and sentiments in text using BERT models.
    
    Built with:
    - FastAPI
    - Streamlit
    - Hugging Face Transformers
    - PyTorch
    """)

# Main header
st.markdown('<p class="header-title">💭 Emotion Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Analyze emotions and sentiments in any text</p>', unsafe_allow_html=True)

# Check API connection
check_api_connection()

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
text_input = st.text_area(
    "Enter text to analyze:",
    value=st.session_state.text_input_value,
    height=150,
    placeholder="Write or paste your text here to analyze its emotions and sentiment..."
)

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])

with col2:
    analyze_button = st.button("🔍 Analyze", 
                              use_container_width=True, 
                              disabled=st.session_state.connection_status=="error")

# Analysis process
if analyze_button and (text_input or uploaded_file):
    # Create a single spinner for the whole process
    with st.spinner("Analyzing text..."):
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
        else:
            text = text_input
            st.session_state.text_input_value = text  # Save last analyzed text
            
        if submit_analysis(text, threshold):
            # Wait for results
            max_retries = 30
            retries = 0
            while retries < max_retries:
                result = get_analysis_result()
                if result and result.get("status") == "completed":
                    st.success("Analysis completed successfully!")
                    
                    # Save result with timestamp
                    result["timestamp"] = format_timestamp()
                    result["text"] = text
                    st.session_state.analysis_history.append(result)
                    
                    # Show results
                    st.subheader("📊 Analysis Results")
                    
                    # Split into columns for sentiment and emotions
                    col_sentiment, col_emotions = st.columns([1, 2])
                    
                    with col_sentiment:
                        # Sentiment card
                        sentiment = result["result"]["sentiment"]
                        sentiment_label = sentiment["label"].upper()
                        sentiment_score = sentiment["score"]
                        
                        # Determine sentiment category
                        sentiment_category = "neutral"
                        if "star" in sentiment_label.lower():
                            stars = sentiment_label.split()[0]
                            if stars in ["1", "2"]:
                                sentiment_category = "negative"
                            elif stars in ["4", "5"]:
                                sentiment_category = "positive"
                        else:
                            if "positive" in sentiment_label.lower():
                                sentiment_category = "positive"
                            elif "negative" in sentiment_label.lower():
                                sentiment_category = "negative"
                        
                        sentiment_color = COLORS.get(sentiment_category, COLORS["neutral_sentiment"])
                        
                        st.markdown(f"""
                        <div class="sentiment-card" style="background-color: {sentiment_color}20; border-left: 5px solid {sentiment_color}">
                            <h3>Detected Sentiment</h3>
                            <h1 style="color: {sentiment_color}; margin: 10px 0;">{sentiment_label}</h1>
                            <p style="font-size: 1.2rem;">Category: <strong>{sentiment_category.upper()}</strong></p>
                            <p style="font-size: 1.2rem;">Confidence: <strong>{sentiment_score:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_emotions:
                        # Fix charts in current analysis results
                        emotions = result["result"]["emotions"]
                        if emotions:
                            # Create bar chart for emotions
                            emotions_fig = create_emotions_bar_chart(emotions)
                            st.plotly_chart(emotions_fig, use_container_width=True, key="current_emotions_chart")
                            
                            # Show number of detected emotions (for diagnostics)
                            st.info(f"Detected emotions: {len(emotions)}")
                        else:
                            st.info("No emotions detected with sufficient confidence.")
                    
                    # Advanced visualization section
                    st.subheader("🔍 Advanced Visualization")
                    
                    # Modify radar chart code
                    if emotions:
                        if len(emotions) >= 3:
                            radar_fig = create_emotion_radar_chart(emotions)
                            st.plotly_chart(radar_fig, use_container_width=True, key="current_radar_chart")
                        else:
                            st.warning(f"At least 3 emotions are needed for the radar chart. Currently have {len(emotions)}.")
                            
                            # Suggest threshold adjustment
                            st.info("Tip: Try lowering the confidence threshold in the sidebar to show more emotions.")
                    
                    break
                elif result and result.get("status") == "error":
                    st.error(f"Analysis error: {result.get('message', 'Unknown error')}")
                    break
                
                time.sleep(1)
                retries += 1
            
            # Show timeout error only at the end
            if retries >= max_retries:
                st.error("Analysis timed out. Please try again.")

# Analysis history
if st.session_state.analysis_history:
    st.divider()
    st.subheader("📜 Analysis History")
    
    # Show last 5 analyses
    for idx, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"Analysis {len(st.session_state.analysis_history) - idx} - {analysis.get('timestamp', 'N/A')}"):
            # Show analyzed text
            st.markdown("**Analyzed text:**")
            st.markdown(f"> _{analysis.get('text', 'N/A')}_")
            
            # Results
            sentiment = analysis["result"]["sentiment"]
            emotions = analysis["result"]["emotions"]
            
            # Split into columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Sentiment:** {sentiment['label'].upper()} ({sentiment['score']:.2f}%)")
                
                # Show emotions in a formatted list
                st.markdown("**Detected emotions:**")
                for emotion in emotions:
                    emotion_color = COLORS.get(emotion['label'].lower(), COLORS['neutral'])
                    st.markdown(
                        f"""<div class="emotion-item">
                            <span class="emotion-label">{emotion['label'].capitalize()}</span>
                            <div class="emotion-bar" style="width: {emotion['score']}px; background-color: {emotion_color};"></div>
                            <span class="emotion-value">{emotion['score']:.2f}%</span>
                        </div>""", 
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Mini chart for this analysis with unique key
                emotions_fig = create_emotions_bar_chart(emotions)
                st.plotly_chart(emotions_fig, use_container_width=True, key=f"history_{idx}_chart")