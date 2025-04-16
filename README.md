# Emotion and Sentiment Analyzer
A Python-based tool that analyzes both sentiment and emotions in text using state-of-the-art BERT models.

## Features
- Real-time sentiment analysis (positive/neutral/negative) with confidence scores
- Detailed emotion analysis with confidence scores and filtering
- Interactive visualization with:
  - Color-coded emotion bar charts
  - Sentiment indicator
  - Configurable display options
- Support for both CPU and GPU processing
- Color-coded terminal output for better readability
- Offline capability (models are saved locally after first run)
- Batch processing support
- Configurable text length limits

## Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (installed via requirements.txt):
  - torch
  - transformers
  - colorama
  - matplotlib
  - tqdm

## Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd emotion_analyzer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
Run the analyzer in interactive mode:
```bash
python emotion_analyzer.py
```

### Command Line Arguments
The following command-line options are available:

- `--no-vis`: Disable visualization of results
- `--batch-size N`: Set the batch size for processing (default: 8)
- `--max-length N`: Set maximum text length (default: 512)

Example with arguments:
```bash
python emotion_analyzer.py --no-vis --batch-size 16 --max-length 256
```

### Interactive Mode
- Enter text when prompted to analyze
- Choose whether to display visualizations
- Press 'q', 'quit', 'exit', or Ctrl+C to exit
- Results show:
  - Sentiment analysis with confidence score
  - Emotion analysis with confidence percentages
  - Visual bar chart (if enabled)
  - Color-coded terminal output

## Models Used
- Sentiment Analysis: `nlptown/bert-base-multilingual-uncased-sentiment`
  - Provides 1-5 star ratings mapped to positive/neutral/negative
- Emotion Analysis: `bhadresh-savani/bert-base-uncased-emotion`
  - Detects multiple emotions with confidence scores
  - Filters out emotions with less than 3% confidence

## Output Format
- Sentiment Analysis shows:
  - Rating (1-5 stars)
  - Simplified category (positive/neutral/negative)
  - Confidence score
- Emotion Analysis shows:
  - Individual emotions with confidence percentages
  - Visual bar chart representation
  - Color-coded visualization

## Error Handling
- Input validation for text length
- Graceful handling of model loading errors
- Proper resource cleanup
- Comprehensive logging

## License
See the [LICENSE](LICENSE) file for details.
