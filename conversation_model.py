import numpy as np

from pre_processing import *

class Recording:
    def __init__(self, file_path, y, sampling_rate):
        self.value = file_path
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
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        # pitch, cadence
    def downsample(self):
        for i, features in enumerate(self.feature_matrix):
            self.feature_matrix[i] = downsample(features)
    def normalize(self):
        for i, features in enumerate(self.feature_matrix):
            self.feature_matrix[i] = normalize(features)
    @property
    def description(self):
        return f"Speaker {self.speaker_id}: '{self.value}'"

# Conversation class used to organize utterances
class Conversation:
    def __init__(self, length, utterances, fidelity):
        self.length = length # in seconds
        self.utterances = utterances # array of utterances
        self.fidelity = fidelity
    @property
    def description(self):
        return f"Speaker {self.speaker_id}: '{self.value}'"