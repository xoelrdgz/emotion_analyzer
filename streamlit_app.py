# frontend/streamlit_app.py
import streamlit as st
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from pathlib import Path

# Page configuration with theme
st.set_page_config(
    page_title="Emotion Analyzer | AI-powered Text Analysis",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_file = Path(__file__).parent / "static" / "style.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css()

# Constants and configuration
DEFAULT_API_URL = "http://localhost:8000"
# Complete emotion mappings based on bhadresh-savani/bert-base-uncased-emotion model
EMOJI_MAP = {
    # Basic emotions
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "sadness": "😢",
    "surprise": "😲",
    "neutral": "😐",
    
    # Secondary emotions
    "worry": "😟",
    "happy": "😄",
    "hate": "😡",
    "admiration": "🤩",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "🤔",
    "curiosity": "🧐",
    "desire": "🥰",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤪",
    "gratitude": "🙏",
    "grief": "😭",
    "nervousness": "😰",
    "optimism": "🌟",
    "pride": "🦁",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "annoyance": "😤",
    
    # Additional nuanced emotions
    "awe": "🥺",
    "anticipation": "🤗",
    "distraction": "🤪",
    "empathy": "💕",
    "enthusiasm": "✨",
    "indifference": "🤷",
}

COLORS = {
    # Sentiment colors
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#f1c40f",
    "neutral_sentiment": "#f1c40f",
    "primary": "#3f51b5",
    "secondary": "#f50057",
    
    # Emotion colors
    "joy": "#2ecc71",
    "love": "#e84393",
    "anger": "#e74c3c",
    "fear": "#8e44ad",
    "sadness": "#3498db",
    "surprise": "#f1c40f",
    "neutral": "#95a5a6",
    "worry": "#e67e22",
    "happy": "#2ecc71",
    "hate": "#c0392b",
    "admiration": "#27ae60",
    "approval": "#16a085",
    "caring": "#9b59b6",
    "confusion": "#34495e",
    "curiosity": "#3498db",
    "desire": "#e74c3c",
    "disappointment": "#95a5a6",
    "disapproval": "#c0392b",
    "disgust": "#d35400",
    "embarrassment": "#e67e22",
    "excitement": "#f1c40f",
    "gratitude": "#27ae60",
    "grief": "#2c3e50",
    "nervousness": "#9b59b6",
    "optimism": "#2ecc71",
    "pride": "#f39c12",
    "realization": "#3498db",
    "relief": "#1abc9c",
    "remorse": "#95a5a6",
    "annoyance": "#e74c3c"
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
            st.error(f"API error: HTTP {response.status_code}")
            return False
    except ConnectionError:
        st.session_state.connection_status = "error"
        st.error("Could not connect to the API server")
        return False
    except Timeout:
        st.session_state.connection_status = "error"
        st.error("API request timed out")
        return False
    except RequestException as e:
        st.session_state.connection_status = "error"
        st.error(f"API request failed: {str(e)}")
        return False

def submit_analysis(text, threshold):
    """Sends text for analysis to the backend"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/analyze",
            json={"text": text, "threshold": threshold / 100},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        st.session_state.job_id = data["job_id"]
        return True
    except ConnectionError:
        st.error("Could not connect to the analysis server")
        return False
    except Timeout:
        st.error("Analysis request timed out")
        return False
    except requests.exceptions.HTTPError as e:
        error_msg = "Analysis request failed"
        try:
            error_data = e.response.json()
            if 'detail' in error_data:
                error_msg = error_data['detail']
        except:
            pass
        st.error(f"{error_msg} (HTTP {e.response.status_code})")
        return False
    except RequestException as e:
        st.error(f"Analysis request failed: {str(e)}")
        return False

def get_analysis_result():
    """Gets analysis results from the backend using SSE"""
    if not st.session_state.job_id:
        return None
    
    placeholder = st.empty()
    
    try:
        with requests.get(
            f"{st.session_state.api_url}/stream/{st.session_state.job_id}",
            stream=True
        ) as response:
            # SSE response processing
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])  # Skip 'data: ' prefix
                        
                        if data.get('status') == 'not_found':
                            st.error("Analysis job not found")
                            return None
                        
                        if data.get('status') in ['completed', 'error']:
                            return data
                        
                        # Update progress message
                        with placeholder:
                            st.write("Analysis in progress...")
                            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_emotion_label_with_emoji(emotion_label: str) -> str:
    """Get formatted emotion label with emoji"""
    emoji = EMOJI_MAP.get(emotion_label.lower(), "")
    return f"{emoji} {emotion_label}"

def create_emotions_bar_chart(emotions):
    """Creates a bar chart for detected emotions"""
    if not emotions:
        return None
    
    # Create dataframe for the chart
    df = pd.DataFrame(emotions)
    df = df.sort_values('score', ascending=True)  # Sort from lowest to highest
    
    # Add emojis to labels
    df['label'] = df['label'].apply(get_emotion_label_with_emoji)
    
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
    
    # Prepare data for radar chart with emojis
    labels = [f"{get_emotion_label_with_emoji(emo['label'])} ({emo['score']:.1f}%)" for emo in emotions]
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

def create_placeholder_radar():
    """Creates a placeholder radar chart when there aren't enough emotions"""
    fig = go.Figure()
    
    # Add a circle to represent the empty radar chart
    theta = [i for i in range(0, 360, 10)]
    r = [1] * len(theta)
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        line=dict(color='rgba(200, 200, 200, 0.3)', dash='dot'),
        fill='toself',
        fillcolor='rgba(200, 200, 200, 0.1)',
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                showline=False,
                ticks='',
            ),
            angularaxis=dict(
                visible=False
            )
        ),
        showlegend=False,
        margin=dict(l=70, r=70, t=20, b=20),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text="Not enough emotions detected<br>for radar visualization",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color='rgba(100, 100, 100, 0.8)')
            )
        ]
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

def display_sentiment_card(sentiment, col):
    """Display sentiment analysis results in a card"""
    with col:
        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]
        sentiment_category = sentiment.get("category", "neutral")  # Get category from new structure
        
        sentiment_color = COLORS.get(sentiment_category, COLORS["neutral_sentiment"])
        
        st.markdown(f"""
        <div class="sentiment-card {sentiment_category}">
            <h3>Detected Sentiment</h3>
            <h1>{sentiment_label}</h1>
            <p>Category: <strong>{sentiment_category.upper()}</strong></p>
            <p>Confidence: <strong>{sentiment_score:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

def display_header():
    """Display page header using native Streamlit components"""
    st.markdown('<p class="header-title">💭 Emotion Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Analyze emotions and sentiments in any text</p>', unsafe_allow_html=True)

def display_connection_status():
    """Display API connection status"""
    if st.session_state.connection_status == "connected":
        status_text = "Connected"
        status_class = "connected"
    elif st.session_state.connection_status == "error":
        status_text = "Disconnected"
        status_class = "error"
    else:
        status_text = "Unknown"
        status_class = "unknown"
    
    st.markdown(
        f'<div>Status: <span class="connection-indicator {status_class}"></span> {status_text}</div>',
        unsafe_allow_html=True
    )

def display_emotion_item(emotion, color):
    """Display a single emotion item with proper styling"""
    emotion_label = get_emotion_label_with_emoji(emotion['label'])
    st.markdown(
        f"""<div class="emotion-item">
            <span class="emotion-label">{emotion_label}</span>
            <div class="emotion-bar" style="width: {emotion['score']}px; background-color: {color};"></div>
            <span class="emotion-value">{emotion['score']:.2f}%</span>
        </div>""",
        unsafe_allow_html=True
    )

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
        display_connection_status()
    
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
display_header()

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

def get_text_input(uploaded_file, text_input):
    """Get text from either file upload or direct input"""
    if uploaded_file:
        return uploaded_file.read().decode("utf-8")
    return text_input

def display_sentiment_card(sentiment, col):
    """Display sentiment analysis results in a card"""
    with col:
        sentiment_label = sentiment["label"]
        sentiment_score = sentiment["score"]
        sentiment_category = sentiment.get("category", "neutral")  # Get category from new structure
        
        sentiment_color = COLORS.get(sentiment_category, COLORS["neutral_sentiment"])
        
        st.markdown(f"""
        <div class="sentiment-card {sentiment_category}">
            <h3>Detected Sentiment</h3>
            <h1>{sentiment_label}</h1>
            <p>Category: <strong>{sentiment_category.upper()}</strong></p>
            <p>Confidence: <strong>{sentiment_score:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

def display_emotions_chart(emotions, col):
    """Display emotions bar chart"""
    with col:
        if emotions:
            emotions_fig = create_emotions_bar_chart(emotions)
            st.plotly_chart(emotions_fig, use_container_width=True, key="current_emotions_chart")
            st.info(f"Detected emotions: {len(emotions)}")
        else:
            st.info("No emotions detected with sufficient confidence.")

def display_radar_chart(emotions):
    """Display radar chart for emotions with improved UX"""
    st.subheader("🔍 Advanced Visualization")
    
    if not emotions:
        with st.container():
            st.markdown("""
            <div class="chart-container" style="opacity: 0.6;">
                <div style="text-align: center; padding: 2rem;">
                    <h3 style="color: #666;">No emotions detected</h3>
                    <p>Try adjusting the confidence threshold in the sidebar to detect more emotions.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return
        
    min_emotions = 3
    current_emotions = len(emotions)
    
    if current_emotions < min_emotions:
        with st.container():
            # Show placeholder chart
            placeholder_fig = create_placeholder_radar()
            st.plotly_chart(placeholder_fig, use_container_width=True)
            
            # Show informative message with progress
            col1, col2 = st.columns([2, 1])
            with col1:
                st.progress(current_emotions / min_emotions, 
                          f"Found {current_emotions}/{min_emotions} emotions needed for radar chart")
            with col2:
                st.button("⚙️ Adjust Threshold", 
                         on_click=lambda: st.sidebar._arrow_container.button("Analysis"))
    else:
        radar_fig = create_emotion_radar_chart(emotions)
        st.plotly_chart(radar_fig, use_container_width=True)

def save_analysis_result(result, text):
    """Save analysis result to history with timestamp"""
    result["timestamp"] = format_timestamp()
    result["text"] = text
    st.session_state.analysis_history.append(result)

def handle_analysis_result(result, text):
    """Process and display analysis results with improved error handling"""
    try:
        if result.get('status') == 'error':
            error_msg = result.get('result', {}).get('error', 'Unknown error')
            st.error(f"Analysis failed: {error_msg}")
            return

        st.success("Analysis completed successfully!")
        
        # Save result to history with error handling
        try:
            save_analysis_result(result, text)
        except Exception as e:
            st.warning(f"Could not save to history: {str(e)}")
        
        # Display results
        st.subheader("📊 Analysis Results")
        
        # Split into columns for sentiment and emotions
        col_sentiment, col_emotions = st.columns([1, 2])
        
        # Display sentiment and emotions with error handling
        try:
            display_sentiment_card(result["result"]["sentiment"], col_sentiment)
            display_emotions_chart(result["result"]["emotions"], col_emotions)
            
            # Advanced visualization section
            st.subheader("🔍 Advanced Visualization")
            display_radar_chart(result["result"]["emotions"])
        except KeyError as e:
            st.error(f"Invalid result format: missing {str(e)}")
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")

    except Exception as e:
        st.error(f"Error processing analysis results: {str(e)}")

def process_text_analysis():
    """Process text analysis when analyze button is clicked"""
    if analyze_button and (text_input or uploaded_file):
        with st.spinner("Analyzing text..."):
            try:
                # Get input text with validation
                text = get_text_input(uploaded_file, text_input)
                if not text or not text.strip():
                    st.error("Please enter some text to analyze")
                    return
                
                if len(text) > 5000:  # Reasonable limit for API
                    st.error("Text is too long (maximum 5000 characters)")
                    return
                
                if text_input:
                    st.session_state.text_input_value = text
                
                # Submit analysis request
                if submit_analysis(text, threshold):
                    # Wait for results using SSE
                    placeholder = st.empty()
                    
                    try:
                        with requests.get(
                            f"{st.session_state.api_url}/stream/{st.session_state.job_id}",
                            stream=True,
                            timeout=30
                        ) as response:
                            response.raise_for_status()
                            
                            for line in response.iter_lines():
                                if line:
                                    line = line.decode('utf-8')
                                    if line.startswith('data: '):
                                        data = json.loads(line[6:])
                                        
                                        if data.get('status') == 'not_found':
                                            st.error("Analysis job not found")
                                            return
                                        
                                        if data.get('status') == 'error':
                                            error_msg = data.get('result', {}).get('error', 'Unknown error')
                                            st.error(f"Analysis error: {error_msg}")
                                            return
                                        
                                        if data.get('status') == 'completed':
                                            handle_analysis_result(data, text)
                                            return
                                        
                                        # Update progress message
                                        with placeholder:
                                            st.write("Analysis in progress...")
                                            
                    except Timeout:
                        st.error("Analysis timed out - please try again")
                    except ConnectionError:
                        st.error("Lost connection to analysis server")
                    except RequestException as e:
                        st.error(f"Error getting analysis results: {str(e)}")
                        
            except UnicodeDecodeError:
                st.error("Could not read the uploaded file - please ensure it's a valid text file")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Analysis process
process_text_analysis()

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