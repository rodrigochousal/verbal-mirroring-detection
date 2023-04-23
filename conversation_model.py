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
    @property
    def length(self):
        return self.end_time-self.start_time
    @property
    def description(self):
        return f"Speaker {self.speaker_id}: '{self.value}'"
    
class UtteranceMatrix:
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

# Conversation class used to organize utterances
class Conversation:
    def __init__(self, length, utterances, window_size):
        self.length = length # in seconds
        self.utterances = utterances # array of utterances
        self.window_size = window_size
        
    def summarize_speakers(self):
        """
        Treat each consecutive utterance from the same speaker as one long, averaged, utterance
        """
        summarized_utterances = []
        s_utterance = self.utterances[0]
        for utterance in self.utterances[1:]:
            if utterance.speaker_id != s_utterance.speaker_id:
                # capture & reset
                summarized_utterances.append(s_utterance)
                s_utterance = utterance
            else:
                # fold in
                n = s_utterance.length/self.window_size
                new_value = (s_utterance.value*n + utterance.value)/(n+1)
                new_end_time = s_utterance.end_time + self.window_size
                folded = Utterance(new_value, s_utterance.speaker_id, s_utterance.start_time, new_end_time)
                s_utterance = folded
        self.utterances = summarized_utterances
    def print_description(self):
        for u in self.utterances:
            if u.speaker_id != -1:
                print(f"Speaker {u.speaker_id}: {u.value:.4f} for {u.length}s")
            else:
                print(f"Silence for {u.length}s")
