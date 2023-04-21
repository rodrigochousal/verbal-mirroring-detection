import numpy as np
import librosa

from conversation_model import *

def extract_rmse(recording, hop_length, frame_length):
    """
    Calculate RMSE for each frame length.
    """
    return librosa.feature.rms(y=recording.y, frame_length=frame_length, hop_length=hop_length, center=True)[0]

# To-do: extract_cadence, extract_pitch, etc.

def downsample(data, window_size):
    """
    Downsample data by taking the mean of consecutive groups of 'window_size' elements.
    """
    downsampled = []
    for i in range(0, len(data), window_size):
        chunk = data[i:i+window_size]
        if chunk: downsampled.append(sum(chunk) / len(chunk))
    return downsampled

def normalize(data):
    """
    This function takes in a list of numerical data and returns a normalized version of the data.
    """
    max_value = max(data)
    min_value = min(data)
    normalized_data = []
    for value in data:
        n_value = (value - min_value) / (max_value - min_value)
        normalized_data.append(n_value)
    return normalized_data

def make_utterances(data, speaker_id, window_size):
    """
    Create Utterance objects from data (optionally downsampled and normalized), 
    with speaker_id and utterance length 'window_size'
    """
    utterances = []
    for i, value in enumerate(data):
        start_time = i*window_size
        end_time = start_time + window_size
        utterance = Utterance(value, speaker_id, start_time, end_time)
        utterances.append(utterance)
    return utterances

def make_conversation(utterance_matrix, length, fidelity):
     # All utterance lists should be the same length, TODO: check this is passed length
    lengths = [len(x) for x in utterance_matrix]
    if not len(set(lengths)) == 1:
       return None
    # Make conversation by choosing speaker feature with greatest value
    max_utterances = []
    for i in range(0, length):
        max_utterance = None
        for jth_utterances in utterance_matrix:
            if jth_utterances[i].value > max_utterance.value:
                max_utterance = jth_utterances[i]
        max_utterances.append(max_utterance)