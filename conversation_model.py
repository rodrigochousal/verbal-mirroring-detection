import numpy as np

from pre_processing import *

class Recording:
    def __init__(self, file_path, y, sampling_rate):
        self.file_path = file_path
        self.y = y
        self.sampling_rate = sampling_rate
    @property
    def description(self):
        sr = f"Sampling Rate: {str(self.sampling_rate)} samples p/s"
        ts = f"Total Samples: {str(np.size(self.y))}"
        length = f"Length: {str(np.size(self.y)/self.sampling_rate)} s"
        return f"{sr}\n{ts}\n{length}\n"

# Utterance class used to identify each time someone speaks
class Utterance:
    def __init__(self, value, speaker_id, start_time, end_time):
        self.value = value # volume
        self.speaker_id = speaker_id
        self.start_time = start_time
        self.end_time = end_time
        self.length = end_time-start_time
        # pitch, cadence
    @property
    def description(self):
        return f"Speaker {self.speaker_id}: '{self.value}'"
    
class UtteranceMatrix:
    # Maybe use this class to do all the pre-processing on a single object?
    def __init__(self, feature_matrix, window_size):
        self.window_size = window_size
        self.utterance_matrix = []
        for i, features in enumerate(feature_matrix):
            utterances = []
            for j, value in enumerate(features):
                start_time = j*self.window_size
                end_time = start_time + self.window_size
                utterance = Utterance(value, i, start_time, end_time)
                utterances.append(utterance)
            self.utterance_matrix.append(utterances)
    def downsample(self):
        for i, utterances in enumerate(self.utterance_matrix):
            self.feature_matrix[i] = downsample(features, self.window_size)
    def normalize(self):
        for i, features in enumerate(self.feature_matrix):
            self.feature_matrix[i] = normalize(features)

# Conversation class used to organize utterances
class Conversation:
    def __init__(self, length, utterances):
        self.length = length # in seconds
        self.utterances = utterances # array of utterances
