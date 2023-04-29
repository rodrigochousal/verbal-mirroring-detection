import os
import numpy as np
import librosa

from conversation_model import *

'''
Pre-processing in data analysis is the process of cleaning and transforming raw data into a format that is suitable for analysis. It involves several steps, including:
Data Cleaning: This step involves identifying and correcting errors, missing values, and inconsistencies in the dataset. The goal is to ensure that the data is accurate, complete, and consistent.
Data Transformation: This step involves converting the data into a format that is suitable for analysis. This may include converting data types, scaling data, or normalizing data.
Data Reduction: This step involves reducing the size of the dataset without losing important information. This may include removing irrelevant variables, identifying outliers, or using dimensionality reduction techniques.
Data Discretization: This step involves converting continuous data into categorical data. This may include binning data into intervals or creating categories based on specific criteria.
Feature Engineering: This step involves creating new features from existing data. This may include creating new variables based on existing variables, or transforming existing variables to create new insights.
'''

# List of valid audio file extensions
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']
# Number of audio samples that are skipped between successive analysis frames. 
# Determines the overlap between adjacent frames and affects the temporal resolution of the analysis.
HOP_LENGTH = 256
# Number of audio samples that are included in each analysis frame. Determines the frequency resolution
# of the analysis and affects the level of detail that can be captured in the audio signal.
FRAME_LENGTH = 512

def get_recordings(audio_paths, start_time, duration):
    recordings = []
    for path in audio_paths:
        # y: amplitude at a specific point in time
        # sr: # of samples of y that are taken per second (Hz)
        y, sr = librosa.core.load(path, offset=start_time, duration=duration)
        recordings.append(Recording(path, y, sr, duration))
    return recordings

def extract_features(args, recordings):
    default_behavior = False
    feature_matrices = {}
    if not (args.volume or args.pitch or args.cadence):
        print("No argument passed, defaulting to analysing volume...")
        default_behavior = True
    if args.volume or default_behavior:
        print("Analysing volume of recordings...")
        rmse_matrix = []
        for r in recordings:
            data = librosa.feature.rms(y=r.y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
            data =  np.array([x if x >= 0.1 else 0 for x in data])
            # data =  np.array([x if x <= 1 else 0 for x in data])
            rmse_matrix.append(data)
        feature_matrices['volume'] = rmse_matrix
    if args.pitch:
        print("Analysing pitch of recordings...")
    if args.cadence:
        print("Analysing cadence of recordings...")
    return feature_matrices

def clean_up(feature_matrices, duration, u_length):
    for key, matrix in feature_matrices.items():
        for i, data in enumerate(matrix):
            rounded_data = []
            for x in data:
                rounded = round(x, 6)
                rounded_data.append(rounded)
            window_size = int(len(data)*u_length/duration)
            downsampled = downsample(data, window_size)
            feature_matrices[key][i] = downsampled
    return feature_matrices

def get_utterance_matrices(feature_matrices, u_length):
    utterance_matrices = {}
    for key, matrix in feature_matrices.items():
        utterance_matrices[key] = UtteranceMatrix(matrix, u_length)
    return utterance_matrices

def get_conversations(utterance_matrices, u_length):
    conversations = []
    for key, matrix_object in utterance_matrices.items():
        matrix = matrix_object.utterance_matrix
        # find the list in matrix with the largest number of elements
        max_length = max(len(list) for list in matrix)
        # create a new utterance list with the elements of the largest list
        for list in matrix:
            if len(list) == max_length:
                loudest_utterances = list
                break
        for list in matrix:
            for i, u in enumerate(list):
                # replace utterance with that of 'loudest' utterance in matrix
                if loudest_utterances[i].value < u.value:
                    loudest_utterances[i] = u
                elif loudest_utterances[i].value == u.value == 0:
                    # capture silence
                    # TODO: What utterance prompts or responds to silence? Is this useful? <---
                    loudest_utterances[i] = Utterance(0, -1, loudest_utterances[i].start_time, loudest_utterances[i].end_time)
        conversation_length = len(loudest_utterances)*u_length
        conversation = Conversation(conversation_length, loudest_utterances, u_length)
        conversation.summarize_speakers()
        conversations.append(conversation)
    return conversations

def enrich_conversations(conversations):
    rich_conversations = []
    for c in conversations:
        rich_utterances = []
        for u in c.utterances:
            if u.speaker_id != -1:
                rich_utterances.append(u)   
        new_length = len(rich_utterances)*c.window_size
        rich_conversation = Conversation(new_length, rich_utterances, c.window_size)
        rich_conversations.append(rich_conversation)
    return rich_conversations

# Helper

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