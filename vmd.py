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
# Determines the size of an utterance frame, adjusts analysis fidelity
DEFAULT_WINDOW_SIZE = 5

# Build Parser object to read in options from command line
args = setup_interface_parser(FEATURE_LABELS)

print("⏳ 1/3 Pre-Processing Data...")

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

print("✅ Finished Pre-Processing Data")
print("⏳ 2/3 Processing Data...")

# For each conversation, perform the requested analysis
for c in rich_conversations:
    if args.p2r:
        speaker_p2r_ratios = prompt_to_response(c)
        for key, ratios in speaker_p2r_ratios.items():
            sum_of_ratios = 0
            for ratio in ratios:
                sum_of_ratios += ratio[0]
            average_ratio = sum_of_ratios/len(ratios)
            print(f"Speaker {key} average P2R ratio: {average_ratio:.4f}")
    if args.r2r:
        speaker_r2r_ratios = response_to_response(c)
        for key, ratios in speaker_r2r_ratios.items():
            sum_of_ratios = 0
            for ratio in ratios:
                sum_of_ratios += ratio[0]
            average_ratio = sum_of_ratios/len(ratios)
            print(f"Speaker {key} average R2R ratio: {average_ratio:.4f}")

print("✅ Finished Processing Data")
print("⏳ 3/3 Post-Processing Data...")

# Rebecca meeting notes:

# Focus on:
    # 1 - Showings results
        # Graphs, scatter plots, scoring, etc.
        # Run program on the entire dataset
        # Come up with some sort of conclusion about people mirroring each other
    # 2 - Implementing more features
        # Use librosa for more feature options
    # 3 - Part of write-up is how to use the code (make it more readable, and include readme for command line)