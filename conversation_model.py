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
    
# Conversation class used to organize utterances
class Conversation:
    def __init__(self, length, utterances, fidelity):
        self.length = length # in seconds
        self.utterances = utterances # array of utterances
        self.fidelity = fidelity
    @property
    def description(self):
        return f"Speaker {self.speaker_id}: '{self.value}'"