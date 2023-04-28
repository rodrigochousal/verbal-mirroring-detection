import os
import numpy as np
import librosa

from conversation_model import *

# List of valid audio file extensions
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']

def get_recordings_from(args):
    # read in the list of file paths from the command line file
    if args.audio_list:
        with open(args.audio_list, "r") as f:
            audio_paths = f.read().splitlines()
    # loop through the file paths and check if they are audio files
    for path in audio_paths:
        # if is file
        if os.path.isfile(path):
            # get the file extension
            ext = os.path.splitext(path)[1].lower()
            # check if the file extension is in the list of valid audio extensions
            if ext not in AUDIO_EXTENSIONS:
                print(f"{path} is not an audio file.")
                raise SystemExit(1)
    recordings = []
    for path in audio_paths:
        # read in the start time and duration for analysis
        start_time = 0
        duration = librosa.get_duration(filename=path)
        if args.start_time:
            start_time = args.start_time
        if args.duration:
            duration = args.duration
        # y: amplitude at a specific point in time
        # sr: # of samples of y that are taken per second (Hz)
        y, sr = librosa.core.load(path, offset=start_time, duration=duration)
        recordings.append(Recording(path, y, sr))
    return recordings

def downsample(data, window_size):
    """
    Downsample data by taking the mean of consecutive groups of 'window_size' non-zero elements.
    """
    downsampled = []
    for i in range(0, len(data), window_size):
        chunk = data[i:i+window_size]
        non_zero_chunk = [i for i in chunk if i > 0]
        if len(non_zero_chunk) > 0:
            downsampled.append(sum(non_zero_chunk) / len(non_zero_chunk))
        else:
            downsampled.append(0)
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

def replace_outliers_zscore(data, threshold):
    """
    Replace outliers in a dataset with the value '0' using the z-score method.

    Parameters:
        data (array-like): The dataset to replace outliers in.
        threshold (float): The z-score threshold above which data points are considered outliers.

    Returns:
        A new dataset with the outliers replaced with '0'.
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    mask = z_scores > threshold
    data[mask] = 0
    return data

def extract_rmse(recording, hop_length, frame_length):
    """
    Calculate RMSE for each frame length.
    """
    return librosa.feature.rms(y=recording.y, frame_length=frame_length, hop_length=hop_length, center=True)[0]

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