import numpy as np
import re
import requests

# 1. Load the Dataset (Shakespeare example) [cite: 8, 17]
url = "https://github.com/tusharsuthar108/Kenya-Data-Train/blob/main/pg100-images-3.txt"
text = requests.get(url).text

# 2. Preprocessing: Lowercase and remove punctuation 
text = text.lower()
text = re.sub(r'[^\w\s]', '', text) # Removes everything except words and spaces

# 3. Create Character Mapping
chars = sorted(list(set(text))) # Get all unique characters
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

n_chars = len(text)
n_vocab = len(chars)
print(f"Total Characters: {n_chars}")
print(f"Unique Characters (Vocab Size): {n_vocab}")

# 4. Prepare Input-Output Pairs 
seq_length = 40 # Number of characters to look at before predicting
step = 3        # Skip some characters to avoid massive data overlap
data_X = []
data_y = []

for i in range(0, n_chars - seq_length, step):
    # Take a chunk of characters (Input) 
    sequence_in = text[i:i + seq_length]
    # Take the very next character (Output) 
    sequence_out = text[i + seq_length]
    
    # Convert characters to integers
    data_X.append([char_to_int[char] for char in sequence_in])
    data_y.append(char_to_int[sequence_out])

n_patterns = len(data_X)
print(f"Total Training Patterns: {n_patterns}")

# 5. Reshape for LSTM [cite: 22]
# LSTM expects input in the shape: [samples, time_steps, features]
X = np.reshape(data_X, (n_patterns, seq_length, 1))
X = X / float(n_vocab) # Normalize data for faster training

# y needs to be "One-Hot Encoded" for Categorical Crossentropy [cite: 24]
from tensorflow.keras.utils import to_categorical
y = to_categorical(data_y)

print("Pre-processing complete. Data is ready for Model Design.")
