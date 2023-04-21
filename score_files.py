import sys
import os
import argparse

import librosa

# project model
from conversation_model import Conversation, Recording
from conversation_model import Utterance

# set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("audio_list", help="path to file containing list of audio file paths")
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

if args.volume:
    print(f"Applying volume effect of {args.volume} to {path}")
if args.pitch:
    print(f"Applying pitch effect of {args.pitch} to {path}")
if args.cadence:
    print(f"Applying cadence effect of {args.cadence} to {path}")

# y: amplitude at a specific point in time
# sr: # of samples of y that are taken per second (Hz)
recordings = []
for path in audio_paths:
    y, sr = librosa.core.load(path, offset=30.0, duration=120.0)
    recordings.append(Recording(path, y, sr))

# calculate features
rmse_p1 = librosa.feature.rms(y=y_p1, frame_length=frame_length, hop_length=hop_length, center=True)[0]
