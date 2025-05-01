# Emotion Analyzer

A Streamlit-based application for analyzing emotions and sentiment in text using pre-trained transformer models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotionanalyzer.streamlit.app/)

## Features

- **Dual Analysis**: Performs both multi-label emotion detection and sentiment analysis.
- **Interactive UI**: User-friendly web interface built with Streamlit.
- **Configurable Threshold**: Adjust the confidence threshold for displaying detected emotions.
- **Analysis History**: View recent analysis results within the session.
- **Visualizations**: Includes bar charts and radar charts for visualizing emotion distribution.
- **File Upload**: Analyze text directly from uploaded `.txt` files.
- **Local Model Loading**: Loads models efficiently from local storage, downloading them if necessary.

## Architecture

- **Frontend/Application**: Streamlit (`streamlit_app.py`)
- **Core Logic**: Custom Python module (`emotion_analyzer.py`)
- **Models**:
  - Emotion Detection: `SamLowe/roberta-base-go_emotions` (fine-tuned RoBERTa)
  - Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment` (fine-tuned BERT)
- **Model Storage**: Models are stored locally in `./emotion_model/` and `./sentiment_model/`.

## Installation (Local)

If you wish to run the application locally:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/xoelrdgz/emotion_analyzer.git
    cd emotion_analyzer
    ```

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have PyTorch installed compatible with your system (CPU/GPU). See [PyTorch installation guide](https://pytorch.org/get-started/locally/).*

## Usage (Local)

1. **Run the Streamlit application:**

    ```bash
    streamlit run streamlit_app.py
    ```

    The application will typically be available at `http://localhost:8501`.

2. **Using the Interface:**
    - Enter text into the text area or upload a `.txt` file.
    - Adjust the emotion confidence threshold using the slider.
    - Click "Analyze" to view sentiment and emotion results.
    - Explore past analyses in the "Analysis History" section.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
