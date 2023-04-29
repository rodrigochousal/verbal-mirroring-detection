from interface import *
from pre_processing import *
from processing import *
from post_processing import *

DEFAULT_U_LENGTH = 1 # 1-second utterances
DEFAULT_START_TIME = 0
DEFAULT_DURATION = 240 # 4 minutes

# Build Parser object to read in options from command line
available_features = ["volume", "pitch", "cadence"]
args = setup_interface_parser(available_features)

# Build AnalysisOptions object to keep track of command line options
options = construct_analysis_options(args)

# Capture options, or use defaults
u_length = DEFAULT_U_LENGTH
start_time = DEFAULT_START_TIME
duration = DEFAULT_DURATION
if args.u_length: u_length = args.u_length
if args.start_time: start_time = args.start_time
if args.duration: duration = args.duration

print("⏳ 1/3 Pre-Processing Data...")

# Build Recording list from arguments passed in command line
recordings = get_recordings(audio_paths, start_time, duration)

# Extract requested features from Recording list as matrix [volume, pitch, cadence, etc.] x [values for each]
feature_matrices = extract_features(args, recordings)

# Clean up data (round, normalize, and downsample)
feature_matrices = clean_up(feature_matrices, u_length, duration)

# Build UtteranceMatrix list from feature matrix list
utterance_matrices = get_utterance_matrices(feature_matrices, u_length)

# Build Conversation list from UtteranceMatrix list; each represents a different feature
conversations = get_conversations(utterance_matrices, u_length)

# print transcription if argument was passed
if args.transcription:
    for c in conversations:
        c.print_description()

# Enrich each Conversation, removing silences
rich_conversations = enrich_conversations(conversations)

print("✅ Finished Pre-Processing Data")
print("⏳ 2/3 Processing Data...")

# For each conversation, perform the requested analysis
analysed_conversations = []
for c in rich_conversations:
    if args.p2r:
        p2r_conversation = prompt_to_response(c)
        analysed_conversations.append(p2r_conversation)
        # speaker_ratios = speaker_p2r_ratios
        # for key, ratios in speaker_p2r_ratios.items():
        #     sum_of_ratios = 0
        #     for ratio in ratios:
        #         sum_of_ratios += ratio[0]
        #     average_ratio = sum_of_ratios/len(ratios)
        #     print(f"Speaker {key} average P2R ratio: {average_ratio:.4f}")
    if args.r2r:
        r2r_conversation = response_to_response(c)
        analysed_conversations.append(r2r_conversation)
        # for key, ratios in speaker_r2r_ratios.items():
        #     sum_of_ratios = 0
        #     for ratio in ratios:
        #         sum_of_ratios += ratio[0]
        #     average_ratio = sum_of_ratios/len(ratios)
        #     print(f"Speaker {key} average R2R ratio: {average_ratio:.4f}")

print("✅ Finished Processing Data")
print("⏳ 3/3 Post-Processing Data...")

for c in analysed_conversations:
    if args.p2r:
        plot_p2r(c)
    if args.r2r:
        plot_r2r(c)

print("✅ Finished Post-Processing Data")