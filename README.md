# Emotion and Sentiment Analyzer 🎭

A sophisticated Python application that performs real-time sentiment and emotion analysis using state-of-the-art BERT models. This tool combines powerful deep learning capabilities with an intuitive interface to analyze text and provide detailed emotional insights.

## 🌟 Key Features

- 📊 **Dual Analysis**
  - Sentiment classification (positive/neutral/negative) with confidence scores
  - Detailed emotion analysis with percentage breakdowns
  - Multi-emotion detection with confidence filtering

- 🎨 **Rich Visualizations**
  - Interactive emotion bar charts
  - Color-coded sentiment indicators
  - Real-time visual feedback
  - Configurable display options

- ⚡ **Performance**
  - GPU acceleration support
  - Batch processing capabilities
  - Configurable text length limits
  - Offline mode with local model storage

- 🛠️ **User Experience**
  - Interactive command-line interface
  - Color-coded terminal output
  - Comprehensive error handling
  - Detailed progress logging

## 📋 Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster processing)

## 📦 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd emotion_analyzer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Command Line Interface

Basic usage:

```bash
python emotion_analyzer.py
```

Available options:

```bash
python emotion_analyzer.py [--no-vis] [--batch-size N] [--max-length N]
```

Arguments:

- `--no-vis`: Disable visualization output
- `--batch-size N`: Set processing batch size (default: 8)
- `--max-length N`: Maximum text length to process (default: 512)

### Web Interface

Start the web application:

```bash
streamlit run streamlit_app.py
```

### API Server

Launch the REST API:

```bash
uvicorn api:app --reload
```

## 🧠 Models

### Sentiment Analysis

- Model: `nlptown/bert-base-multilingual-uncased-sentiment`
- Features:
  - 5-star rating system
  - Mapped to positive/neutral/negative categories
  - Confidence scoring

### Emotion Analysis

- Model: `bhadresh-savani/bert-base-uncased-emotion`
- Features:
  - Multi-emotion detection
  - Confidence thresholding (3% minimum)
  - Detailed emotion breakdown

## 📊 Output Format

### Sentiment Results

- Rating classification
- Sentiment category
- Confidence percentage
- Color-coded indicators

### Emotion Results

- Multiple emotion detection
- Confidence percentages
- Visual bar charts
- Color-coded output

## 🔧 Advanced Features

- Batch text processing
- GPU acceleration
- Local model caching
- Configurable visualization
- Comprehensive logging
- Error handling and recovery

## 📄 License

This project is licensed under the terms of the included LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

For detailed API documentation and integration examples, see the `api.py` and `service.py` files.
