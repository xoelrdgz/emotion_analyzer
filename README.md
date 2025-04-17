# Emotion Analyzer

A natural language processing application that performs emotion detection and sentiment analysis on text using state-of-the-art BERT models. The application provides both a RESTful API and a user-friendly web interface.

## Features

- **Dual Analysis**: Performs both emotion detection and sentiment analysis on text
- **Real-time Processing**: Asynchronous processing with live updates via Server-Sent Events (SSE)
- **Multi-language Support**: Sentiment analysis supports multiple languages
- **Interactive UI**: Modern web interface with real-time visualizations
- **RESTful API**: Well-documented API endpoints for integration
- **Rich Visualizations**: Interactive charts and graphs for emotion/sentiment distribution

## Architecture

- **Frontend**: Streamlit-based web application
- **Backend**: FastAPI-based REST API
- **Models**: Fine-tuned BERT models
  - Emotion Detection: `bert-base-uncased-emotion`
  - Sentiment Analysis: `bert-base-multilingual-uncased-sentiment`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/emotion_analyzer.git
cd emotion_analyzer
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
python main.py
```

The API server will start on `http://localhost:8000`. API documentation is available at `/docs`.

### Starting the Web Interface

```bash
streamlit run streamlit_app.py
```

The web interface will be available at `http://localhost:8501`.

### Training Models (Optional)

To train the models from scratch:

```bash
# Train emotion detection model
python train_emotion_model.py --epochs 3 --batch_size 16 --learning_rate 2e-5

# Train sentiment analysis model
python train_sentiment_model.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

## API Endpoints

- `POST /analyze`: Submit text for analysis
  - Parameters:
    - `text`: Input text to analyze
    - `threshold`: Confidence threshold (0-1, default: 0.03)
  - Returns: Job ID for tracking analysis progress

- `GET /stream/{job_id}`: Stream analysis results
  - Returns: Server-Sent Events with analysis progress and results

- `GET /result/{job_id}`: Get analysis results
  - Returns: Complete analysis results if available

## Response Format

```json
{
  "sentiment": {
    "label": "POSITIVE",
    "score": 85.5,
    "category": "positive"
  },
  "emotions": [
    {
      "label": "joy",
      "score": 75.2
    },
    {
      "label": "love",
      "score": 24.8
    }
  ]
}
```

## Project Structure

- `api.py`: FastAPI application and endpoints
- `emotion_analyzer.py`: Core analysis engine
- `service.py`: Service layer managing model resources
- `streamlit_app.py`: Web interface implementation
- `exceptions.py`: Custom exception hierarchy
- `train_*.py`: Model training scripts
- `requirements.txt`: Project dependencies
- `static/`: Static assets (CSS, etc.)
- `*_model/`: Trained model directories

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Streamlit
- See `requirements.txt` for complete list

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## Acknowledgments

- Emotion detection model: [bhadresh-savani/bert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion)
- Sentiment analysis model: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
