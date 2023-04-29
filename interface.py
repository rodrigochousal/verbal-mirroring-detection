import argparse

def setup_interface_parser(feature_labels):
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
    for label in feature_labels:
        parser.add_argument(f"--{label}", help=f"analyze {label} mirroring in the conversation", action='store_true')
    parser.add_argument("--transcription", help="print a transcription of speaker, value, length for each utterance in conversation", action='store_true')
    parser.add_argument("--r2r", help="score speakers on response to response", action='store_true')
    parser.add_argument("--p2r", help="score speakers on prompt to response", action='store_true')
    return parser.parse_args()