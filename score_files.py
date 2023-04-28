import sys
import os
import argparse
import numpy as np
import librosa

# project
from interface import *
from pre_processing import *
from processing import *

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

# Build Parser object to read in options from command line
args = setup_interface_parser(FEATURE_LABELS)

print("Pre-Processing Data...")

# Build Recording list from arguments passed in command line
recordings = get_recordings(args)

# Extract requested features from Recording list as matrix [volume, pitch, cadence, etc.] x [values for each]
feature_matrices = extract_features(args, recordings)

# Clean up data (round, normalize, and downsample)
# read in the window size for analysing audio files
window_size = DEFAULT_WINDOW_SIZE
if args.window_size: window_size = args.window_size
feature_matrices = clean_up(feature_matrices, window_size)

# Build UtteranceMatrix list from feature matrix list
utterance_matrices = get_utterance_matrices(feature_matrices, window_size)

# Build Conversation list from UtteranceMatrix list; each represents a different feature
conversations = get_conversations(utterance_matrices, window_size)

# print transcription if argument was passed
if args.transcription:
    for c in conversations:
        c.print_description()

# Enrich each Conversation, removing silences
rich_conversations = enrich_conversations(conversations)

print("Processing Data...")

# For each conversation, perform the requested analysis
for c in rich_conversations:
    if args.p2r:
        prompt_to_response(c)
    if args.r2r:
        response_to_response(c)

# Rebecca meeting notes:

# Focus on:
    # 1 - Showings results
        # Incorporate some stats magic to get rid of outliers
        # Graphs, scatter plots, scoring, etc.
        # Run program on the entire dataset
        # Come up with some sort of conclusion about people mirroring each other
    # 2 - Implementing more features
        # Use librosa for more feature options
    # 3 - Part of write-up is how to use the code (make it more readable, and include readme for command line)