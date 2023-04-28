import argparse

def setup_parser():
    # set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze some recordings.')
    parser.add_argument("audio_list", help="path to file containing list of audio file paths")
    parser.add_argument("--window_size", type=int, help="size of utterance frame, adjusts analysis fidelity (default 5s)")
    parser.add_argument("--start_time", type=int, help="start time in seconds (default min)")
    parser.add_argument("--duration", type=int, help="duration in seconds (default max)")
    for label in FEATURE_LABELS:
        parser.add_argument(f"--{label}", help=f"analyze {label} mirroring in the conversation", action='store_true')
    parser.add_argument("--transcription", help="print a transcription of speaker, value, length for each utterance in conversation", action='store_true')
    parser.add_argument("--r2r", help="score speakers on response to response", action='store_true')
    parser.add_argument("--p2r", help="score speakers on prompt to response", action='store_true')
    args = parser.parse_args()