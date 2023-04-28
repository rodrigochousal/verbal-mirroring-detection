import sys
import os
import argparse
import numpy as np
import librosa

# project
from interface import *
from conversation_model import *
from pre_processing import *

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
args = setup_parser(FEATURE_LABELS)

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
conversations = get_conversation(utterance_matrices, window_size)

# print transcription if argument was passed
if args.transcription:
    for c in conversations:
        c.print_description()

# calculate scores for each speaker

# TODO: These could probably more easily be properties 
# (relative change of utterance value compared to previous speaker's utterance)

# analyze conversation w/o silences (significant utterances)
sig_utterances = []
for u in conversation.utterances:
    if u.speaker_id != -1:
        sig_utterances.append(u)

if args.p2r:
    # Calculate average feature ratio of reply:prompt for each utterance
    speaker_p2r_ratios = {}
    # For each utterance
    for i, u in enumerate(sig_utterances):
        # Ignore silence or first utterance
        if (u.speaker_id == -1) or (i == 0): continue
        # Find previous non-zero utterance
        prev_nz_value = -1
        for j in range(i-1, 0, -1):
            jth_value = sig_utterances[j].value
            if (jth_value > 0):
                prev_nz_value = jth_value
                break
        if (prev_nz_value == -1): continue
        # Calculate average feature ratio for reply:prompt non-zero utterance
        sid = u.speaker_id
        p2r = prev_nz_value/u.value
        if sid in speaker_p2r_ratios:
            speaker_p2r_ratios[sid].append((p2r, u.length))
        else:
            speaker_p2r_ratios[sid] = [(p2r, u.length)]
    for key, ratios in speaker_p2r_ratios.items():
        sum_of_ratios = 0
        for ratio in ratios:
            sum_of_ratios += ratio[0]
        average_ratio = sum_of_ratios/len(ratios)
        print(f"Speaker {key} average P2R ratio: {average_ratio:.4f}")
if args.r2r:
    # Calculate relative change in average feature value for same speaker's responses
    # compared to the prompter's relative change
    speaker_r2r_ratios = {}
    conversation_rr_ratios = []
    # For each significant utterance
    for i, u in enumerate(sig_utterances):
        # Ignore first and second utterances
        if (i < 3): continue
        # Find same speaker's previous utterance
        sid = u.speaker_id
        curr_value_0 = u.value
        prev_value_0 = -1 # this could be u.prev.value
        for j in range(i-1, 0, -1):
            jth_value = sig_utterances[j].value
            jth_id = sig_utterances[j].speaker_id
            if (jth_id == sid):
                prev_value_0 = jth_value
                break
        if (prev_value_0 == -1.0): continue
        # Find different speaker's previous utterance
        prompt_utterance = sig_utterances[i-1]
        sid = prompt_utterance.speaker_id
        curr_value_1 = prompt_utterance.value
        prev_value_1 = -1 # this could be u.prev.value
        for j in range(i-2, 0, -1):
            jth_value = sig_utterances[j].value
            jth_id = sig_utterances[j].speaker_id
            if (jth_id == sid):
                prev_value_1 = jth_value
                break
        if (prev_value_1 == -1.0): continue
        # Calculate r2r
        speaker_change = curr_value_0/prev_value_0
        prompter_change = curr_value_1/prev_value_1
        r2r = speaker_change/prompter_change
        if sid in speaker_r2r_ratios:
            speaker_r2r_ratios[sid].append((r2r, u.length))
        else:
            speaker_r2r_ratios[sid] = [(r2r, u.length)]
    for key, ratios in speaker_r2r_ratios.items():
        sum_of_ratios = 0
        for ratio in ratios:
            sum_of_ratios += ratio[0]
        average_ratio = sum_of_ratios/len(ratios)
        print(f"Speaker {key} average R2R ratio: {average_ratio:.4f}")


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