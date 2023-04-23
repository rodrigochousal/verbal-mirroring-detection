import sys
import os
import argparse

import librosa

# project
from conversation_model import *
from pre_processing import *

# List of valid audio file extensions
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']
# List of available features for analysis
FEATURE_LABELS = ["volume", "pitch", "cadence"]
# Number of audio samples that are skipped between successive analysis frames. 
# Determines the overlap between adjacent frames and affects the temporal resolution of the analysis.
HOP_LENGTH = 256
# Number of audio samples that are included in each analysis frame. Determines the frequency resolution
# of the analysis and affects the level of detail that can be captured in the audio signal.
FRAME_LENGTH = 512
# Determines the size of an utterance frame, adjusts analysis fidelity
DEFAULT_WINDOW_SIZE = 5

# set up command line arguments
parser = argparse.ArgumentParser(description='Analyze some recordings.')
parser.add_argument("audio_list", help="path to file containing list of audio file paths")
parser.add_argument("--window_size", type=int, help="size of utterance frame, adjusts analysis fidelity (default 5s)")
for label in FEATURE_LABELS:
    parser.add_argument(f"--{label}", help=f"analyze {label} mirroring in the conversation", action='store_true')
args = parser.parse_args()

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
    # y: amplitude at a specific point in time
    # sr: # of samples of y that are taken per second (Hz)
    y, sr = librosa.core.load(path, offset=30.0, duration=120.0)
    recordings.append(Recording(path, y, sr))

# read in the window size for analysing audio files
if args.window_size:
    window_size = args.window_size
else:
    window_size = DEFAULT_WINDOW_SIZE

# extract desired features
default_behavior = False
feature_matrices = {}
if not (args.volume or args.pitch or args.cadence):
    print("No argument passed, defaulting to analysing volume...")
    default_behavior = True
if args.volume or default_behavior:
    print("Analysing volume of recordings...")
    rmse_matrix = []
    for r in recordings:
        rmse_matrix.append(librosa.feature.rms(y=r.y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0])
    feature_matrices['volume'] = rmse_matrix
if args.pitch:
    print("Analysing pitch of recordings...")
if args.cadence:
    print("Analysing cadence of recordings...")

# pre-process feature matrices
for key, matrix in feature_matrices.items():
    for i, data in enumerate(matrix):
        # downsample & normalize each data list in each feature matrix
        feature_matrices[key][i] = downsample(normalize(data), window_size)

# convert feature matrices into utterance matrices
utterance_matrices = {}
for key, matrix in feature_matrices.items():
    utterance_matrices[key] = UtteranceMatrix(matrix, window_size)

# make conversations from utterance matrices
conversations = []
for key, matrix_object in utterance_matrices.items():
    matrix = matrix_object.utterance_matrix
    # find the list in matrix with the largest number of elements
    max_length = max(len(list) for list in matrix)
    # create a new utterance list with the elements of the largest list
    loudest_utterances = [element for list in matrix for element in list if len(list) == max_length]
    for list in matrix:
        for i, u in enumerate(list):
            # replace utterance with that of 'loudest' utterance in matrix
            if loudest_utterances[i].value < u.value:
                loudest_utterances[i] = u
    conversation_length = len(loudest_utterances)*window_size
    conversation = Conversation(conversation_length, loudest_utterances, window_size)
    conversation.summarize_speakers()
    conversations.append(conversation)

for c in conversations:
    c.print_description()

# calculate mirroring score for each speaker in each conversation