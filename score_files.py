import sys
import os
import argparse

import librosa

# project
from conversation_model import *
from pre_processing import *

# set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("audio_list", help="path to file containing list of audio file paths")
parser.add_argument("--window_size", type=int, help="size of frame for each utterance, adjusts fidelity of the analysis (default 5s)")
parser.add_argument("--volume", type=int, help="analyze volume mirroring in the conversation")
parser.add_argument("--pitch", type=int, help="analyze pitch mirroring in the conversation")
parser.add_argument("--cadence", type=int, help="analyze cadence mirroring in the conversation")
args = parser.parse_args()

# list of valid audio file extensions
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']

# read in the list of file paths from the command line file
with open(args.file_list, "r") as f:
    audio_paths = f.read().splitlines()

# loop through the file paths and check if they are audio files
for path in audio_paths:
    # If isfile
    if os.path.isfile(path):
        # get the file extension
        ext = os.path.splitext(path)[1].lower()
         # check if the file extension is in the list of valid audio extensions
        if ext not in AUDIO_EXTENSIONS:
            print(f"{path} is not an audio file.")
            raise SystemExit(1)

# y: amplitude at a specific point in time
# sr: # of samples of y that are taken per second (Hz)
recordings = []
for path in audio_paths:
    y, sr = librosa.core.load(path, offset=30.0, duration=120.0)
    recordings.append(Recording(path, y, sr))

# Hop length refers to the number of audio samples that are skipped between successive analysis 
# frames in a digital audio signal processing operation. It determines the overlap between adjacent 
# frames and affects the temporal resolution of the analysis.
hop_length = 256
# Frame length refers to the number of audio samples that are included in each analysis frame in a 
# digital audio signal processing operation. It determines the frequency resolution of the analysis 
# and affects the level of detail that can be captured in the audio signal.
frame_length = 512
# Determines the fidelity of analysis. Features are averaged every 'window_size' seconds.
window_size = 5

# extract desired features
feature_matrices = [] # this should probably be a dictionary with volume:matrix, pitch:matrix, cadence:matrix
if args.window_size:
    window_size = args.window_size
if args.volume:
    rmse_matrix = []
    for r in recordings:
        rmse_matrix.append(librosa.feature.rms(y=r.y, frame_length=frame_length, hop_length=hop_length, center=True)[0])
    feature_matrices.append(rmse_matrix)
if args.pitch:
    print(f"Applying pitch effect of {args.pitch} to {path}")

# pre-process feature matrices
for i, fm in enumerate(feature_matrices):
    feature_matrices[i] = downsample(normalize(fm), window_size)

# convert feature matrices into utterance matrices
utterance_matrices = []
for fm in feature_matrices:
    utterance_matrices.append(UtteranceMatrix(fm, window_size))

# make conversations from utterance matrices
conversations = []
for matrix in utterance_matrices:
    # find the list in matrix with the largest number of elements
    max_length = max(len(l) for l in matrix)
    # create a new utterance list with the elements of the largest list
    loudest_utterances = [element for sublist in matrix for element in sublist if len(sublist) == max_length]
    for utterance_list in matrix:
        for i, utterance in enumerate(utterance_list):
            # replace utterance with that of 'loudest' utterance in matrix
            if loudest_utterances[i].value < utterance.value:
                loudest_utterances[i] = utterance
    conversation_length = len(loudest_utterances)*window_size
    conversations.append(Conversation(conversation_length, loudest_utterances))

# calculate mirroring score for each speaker in each conversation