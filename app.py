from flask import Flask, render_template, request, jsonify
import re
import nltk
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import WittenBellInterpolated
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from werkzeug.utils import secure_filename
import threading
import time
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}
app.config['FILES_UPLOADED'] = False
app.config['FILE_DELETION_DELAY'] = 300  # 5 minutes in seconds

# Initialize NLTK
nltk.download('punkt', quiet=True)

# Model configuration
MODEL_CONFIG = {
    'n': 4,
    'oov_token': "<UNK>",
    'pad_symbol': "<s>"
}

# Global model variables
model = None
training_data = []
testing_data = []
unique_vocab = set()
uploaded_files = {}  # Track uploaded files and their deletion times

# Regex patterns
CLEAN_TEXT_PATTERN = re.compile(r"[^\w\s]")
REMOVE_BRACKETS_PATTERN = re.compile(r"\[.?\]|\{.?\}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    """Lowercases, removes special characters, and tokenizes."""
    if not text:
        return []
    text = REMOVE_BRACKETS_PATTERN.sub("", text)
    text = CLEAN_TEXT_PATTERN.sub("", text).lower()
    return word_tokenize(text)

def load_text(file_path):
    """Loads and preprocesses text from a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            return preprocess_text(content)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default token.")
        return [MODEL_CONFIG['oov_token']]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [MODEL_CONFIG['oov_token']]

def initialize_model(train_path=None, test_path=None):
    """Initialize the n-gram model with training data."""
    global model, training_data, testing_data, unique_vocab
    
    # Use uploaded files if available, otherwise default paths
    train_path = train_path or os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt')
    test_path = test_path or os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt')
    
    # Fall back to original files if uploads don't exist
    if not os.path.exists(train_path):
        train_path = "train.txt"
    if not os.path.exists(test_path):
        test_path = "test.txt"
    
    # Load and preprocess training data
    training_data = load_text(train_path)
    
    # Handle OOV
    unique_vocab = set(training_data)
    training_data = [
        word if word in unique_vocab else MODEL_CONFIG['oov_token'] 
        for word in training_data
    ]
    
    # Pad sequence only if we have data
    if training_data:
        training_data = list(pad_sequence(
            training_data, 
            MODEL_CONFIG['n'], 
            pad_left=True, 
            left_pad_symbol=MODEL_CONFIG['pad_symbol']
        ))
    else:
        training_data = [MODEL_CONFIG['pad_symbol'] * MODEL_CONFIG['n']]
    
    # Generate ngrams and train model
    ngrams_list = list(everygrams(training_data, max_len=MODEL_CONFIG['n']))
    model = WittenBellInterpolated(MODEL_CONFIG['n'])
    if training_data:
        model.fit([ngrams_list], vocabulary_text=training_data)
    
    # Load and preprocess testing data
    testing_data = load_text(test_path)
    testing_data = [
        word if word in unique_vocab else MODEL_CONFIG['oov_token'] 
        for word in testing_data
    ]
    
    # Pad sequence only if we have data
    if testing_data:
        testing_data = list(pad_sequence(
            testing_data, 
            MODEL_CONFIG['n'], 
            pad_left=True, 
            left_pad_symbol=MODEL_CONFIG['pad_symbol']
        ))
    else:
        testing_data = [MODEL_CONFIG['pad_symbol'] * MODEL_CONFIG['n']]

def schedule_file_deletion(filepath):
    """Schedule a file for deletion after the configured delay."""
    deletion_time = datetime.now() + timedelta(seconds=app.config['FILE_DELETION_DELAY'])
    uploaded_files[filepath] = deletion_time
    print(f"Scheduled deletion for {filepath} at {deletion_time}")

def cleanup_files():
    """Periodically check and delete expired files."""
    while True:
        now = datetime.now()
        to_delete = []
        
        for filepath, deletion_time in uploaded_files.items():
            if now >= deletion_time and os.path.exists(filepath):
                to_delete.append(filepath)
        
        for filepath in to_delete:
            try:
                os.remove(filepath)
                del uploaded_files[filepath]
                print(f"Deleted file: {filepath}")
            except Exception as e:
                print(f"Error deleting file {filepath}: {e}")
        
        time.sleep(60)  # Check every minute

# Start the cleanup thread when the app starts
cleanup_thread = threading.Thread(target=cleanup_files)
cleanup_thread.daemon = True
cleanup_thread.start()

@app.route('/')
def index():
    """Render the main page with conditional content."""
    # Check if files exist in upload folder
    train_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'))
    test_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt'))
    files_uploaded = train_exists and test_exists
    
    if not files_uploaded:
        return render_template('index.html', files_uploaded=False)
    
    # Only initialize model and process data if files exist
    if not model:
        initialize_model()
    
    # Compute scores for test data - handle empty case
    scores = []
    if len(testing_data) >= MODEL_CONFIG['n']:
        try:
            scores = [
                model.score(
                    testing_data[i + MODEL_CONFIG['n'] - 1], 
                    testing_data[i : i + MODEL_CONFIG['n'] - 1]
                ) 
                for i in range(len(testing_data) - (MODEL_CONFIG['n'] - 1))
            ]
        except IndexError:
            scores = [0.0]  # Default value if computation fails
    
    scores_np = np.array(scores) if scores else np.array([0.0])
    
    # Compute perplexity with safeguard for empty data
    perplexity = 0.0
    if len(scores_np) > 0:
        try:
            perplexity = np.exp(-np.sum(np.log(scores_np + 1e-10)) / len(scores_np))
        except:
            perplexity = 0.0
    
    # Prepare heatmap data with safeguards
    width = 8
    height = max(1, np.ceil(len(testing_data) / width).astype(int))
    padded_scores = np.zeros(width * height)
    
    if len(scores_np) > 0:
        padded_scores[:min(len(scores_np), len(padded_scores))] = scores_np[:len(padded_scores)]
    
    diff = max(0, len(padded_scores) - len(scores_np))
    smoothed_scores = gaussian_filter(padded_scores, sigma=1.0).reshape(height, width)
    
    # Format labels with safeguards
    labels = []
    if len(testing_data) >= MODEL_CONFIG['n']:
        labels = [
            " ".join(testing_data[i : i + width]) 
            for i in range(MODEL_CONFIG['n'] - 1, len(testing_data), width)
        ]
    else:
        labels = [" ".join(testing_data)] if testing_data else [""]
    
    labels_individual = [x.split() for x in labels] if labels else [[]]
    
    # Only try to pad last row if we have rows and need padding
    if labels_individual and diff > 0:
        if not labels_individual[-1]:
            labels_individual[-1] = [""] * width
        else:
            labels_individual[-1] += [""] * diff
    
    # Prepare data for template
    heatmap_data = {
        'scores': smoothed_scores.tolist(),
        'labels': labels,
        'individual_labels': labels_individual,
        'stats': {
            'perplexity': f"{perplexity:.4f}",
            'vocab_size': len(model.vocab) if model else 0,
            'ngram_count': len(list(everygrams(training_data, max_len=MODEL_CONFIG['n']))) if training_data else 0,
            'test_length': len(testing_data),
            'n_value': MODEL_CONFIG['n']
        }
    }
    
    return render_template('index.html', 
                         files_uploaded=True,
                         heatmap_data=heatmap_data)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    if 'train_file' not in request.files or 'test_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    train_file = request.files['train_file']
    test_file = request.files['test_file']
    
    if train_file.filename == '' or test_file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if train_file and allowed_file(train_file.filename) and test_file and allowed_file(test_file.filename):
        try:
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save files
            train_filename = secure_filename('train.txt')
            test_filename = secure_filename('test.txt')
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], train_filename)
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
            
            train_file.save(train_path)
            test_file.save(test_path)
            
            # Schedule files for deletion
            schedule_file_deletion(train_path)
            schedule_file_deletion(test_path)
            
            # Reinitialize model with new files
            initialize_model(train_path, test_path)
            app.config['FILES_UPLOADED'] = True
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/lookup', methods=['POST'])
def lookup():
    """Handle probability lookup requests."""
    if not model:
        initialize_model()
    
    data = request.json
    word = data.get('word', '').strip()
    context = data.get('context', '').strip().split()
    
    if not word:
        return jsonify({
            'success': False,
            'error': 'Word cannot be empty'
        })
    
    if word not in model.vocab:
        word = MODEL_CONFIG['oov_token']
    
    context_tokens = [
        w if w in model.vocab else MODEL_CONFIG['oov_token'] 
        for w in context
    ]
    
    try:
        # Ensure context has the right length (n-1)
        if len(context_tokens) > MODEL_CONFIG['n'] - 1:
            context_tokens = context_tokens[-(MODEL_CONFIG['n'] - 1):]
        elif len(context_tokens) < MODEL_CONFIG['n'] - 1:
            context_tokens = [MODEL_CONFIG['pad_symbol']] * (MODEL_CONFIG['n'] - 1 - len(context_tokens)) + context_tokens
        
        probability = model.score(word, context_tokens)
        return jsonify({
            'success': True,
            'result': {
                'word': word,
                'context': ' '.join(context_tokens),
                'probability': f"{probability:.6f}"
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Check if files already exist in upload folder
    train_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'))
    test_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt'))
    app.config['FILES_UPLOADED'] = train_exists and test_exists
    
    if app.config['FILES_UPLOADED']:
        initialize_model()
    
    app.run(debug=True)