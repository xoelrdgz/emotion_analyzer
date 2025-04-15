# Emotion Analyzer

A command-line tool that analyzes both sentiment and emotions in text using state-of-the-art machine learning models.

## Features

- **Sentiment Analysis**: Determines if text is positive, negative, or neutral with confidence scores
- **Emotion Analysis**: Detects 28 different emotions including joy, sadness, anger, fear, etc.
- **Color-coded Output**: Results are displayed with color coding for better readability
- **Multi-lingual Support**: Can analyze text in multiple languages (for sentiment analysis)

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Colorama

## Installation

1. Clone this repository:
```bash
git clone https://github.com/xoelrdgz/emotion_analyzer.git
cd emotion_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer:

```bash
python emotion_analyzer.py
```

Enter text when prompted and press Enter. The tool will display:
- Sentiment analysis (Positive/Negative/Neutral) with confidence score
- Top 5 detected emotions with their confidence scores

Enter 'q' to quit the program.

## Models

The project uses two transformer models:
- Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment`
- Emotion Analysis: `bhadresh-savani/bert-base-uncased-emotion`

## License

MIT License - see LICENSE file for details