import os
import argparse

AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']
SUPPORTED_FEATURES = ["volume", "pitch", "cadence"]

DEFAULT_U_LENGTH = 3 # 3-second utterances
DEFAULT_START_TIME = 0
DEFAULT_DURATION = 240 # 4 minutes

class AnalysisOptions:
    def __init__(self, file_paths, u_length, start_time, duration, 
                 requested_features, transcription_on, requested_analyses):
        self.file_paths = file_paths
        self.u_length = u_length
        self.start_time = start_time
        self.duration = duration
        self.requested_features = requested_features
        self.transcription_on = transcription_on
        self.requested_analyses = requested_analyses

def setup_interface_parser():
    '''
    Set up argument parser for a command-line interface to analyze recordings of conversations.

    Args:
    feature_labels (list): A list of feature labels to analyze.

    Returns:
    parser.parse_args(): Parsed command-line arguments.
    '''
    # set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze some recordings.')
    parser.add_argument("audio_list", help="path to file containing list of audio file paths")
    parser.add_argument("--u_length", type=int, help="length of utterance in sec, adjusts analysis fidelity (default 2s)")
    parser.add_argument("--start_time", type=int, help="start time in seconds (default min)")
    parser.add_argument("--duration", type=int, help="duration in seconds (default max)")
    for label in SUPPORTED_FEATURES:
        parser.add_argument(f"--{label}", help=f"analyze {label} mirroring in the conversation", action='store_true')
    parser.add_argument("--transcription", help="print a transcription of speaker, value, length for each utterance in conversation", action='store_true')
    parser.add_argument("--r2r", help="score speakers on response to response", action='store_true')
    parser.add_argument("--p2r", help="score speakers on prompt to response", action='store_true')
    return parser.parse_args()

def construct_analysis_options(args):
    audio_paths = []
    u_length = DEFAULT_U_LENGTH
    start_time = DEFAULT_START_TIME
    duration = DEFAULT_DURATION
    requested_features = {}
    transcription_on = False
    requested_analyses = {}
    if args.audio_list:
        audio_paths = parse_audio_paths(args.audio_list, AUDIO_EXTENSIONS)
    if args.u_length:
        u_length = args.u_length
    if args.start_time:
        start_time = args.start_time
    if args.duration:
        duration = args.duration
    if args.volume:
        requested_features['volume'] = True
    if args.pitch:
        requested_features['pitch'] = True
    if args.cadence:
        requested_features['cadence'] = True
    if not bool(requested_features):
        requested_features['volume'] = True
    if args.transcription:
        transcription_on = True
    if args.p2r:
        requested_analyses['p2r'] = True
    if args.r2r:
        requested_analyses['r2r'] = True
    if not bool(requested_analyses):
        requested_analyses['p2r'] = True
    return AnalysisOptions(audio_paths, u_length, start_time, duration, requested_features, 
                           transcription_on, requested_analyses)

# Helpers

def parse_audio_paths(audio_list, allowed_extensions):
    # Capture audio paths
    with open(audio_list, "r") as f:
        audio_paths = f.read().splitlines()
    # loop through the file paths and check if they are audio files
    for path in audio_paths:
        if os.path.isfile(path):
            # get the file extension
            ext = os.path.splitext(path)[1].lower()
            # check if the file extension is in the list of valid audio extensions
            if ext not in allowed_extensions:
                print(f"{path} is not an audio file.")
                raise SystemExit(1)
    return audio_paths