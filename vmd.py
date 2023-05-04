import interface
import pre_processing
import processing
import post_processing

# Build Parser object to read in options from command line
args = interface.setup_interface_parser()

# Build AnalysisOptions object to keep track of command line options
options = interface.construct_analysis_options(args)

print("⏳ 1/3 Pre-Processing Data...")

# Build Recording list from arguments passed in command line
recordings = pre_processing.get_recordings(options)

# Extract requested features from Recording list as matrix [volume, pitch, cadence, etc.] x [values for each]
feature_matrices = pre_processing.extract_features(options, recordings)

# Clean up data (round, normalize, and downsample)
feature_matrices = pre_processing.clean_up(options, feature_matrices)

# Build UtteranceMatrix list from feature matrix list
utterance_matrices = pre_processing.get_utterance_matrices(options, feature_matrices)

# Build Conversation list from UtteranceMatrix list; each represents a different feature
conversations = pre_processing.get_conversations(options, utterance_matrices)

# print transcription if argument was passed
if options.transcription_on:
    for c in conversations:
        c.print_description()

# Enrich each Conversation, removing silences
rich_conversations = pre_processing.enrich_conversations(conversations)

print("✅ Finished Pre-Processing Data")
print("----------------------------")
print("⏳ 2/3 Processing Data...")

# For each conversation, perform the requested analysis
analysed_conversations = processing.extract_analyses(options, rich_conversations)

print("✅ Finished Processing Data")
print("----------------------------")
print("⏳ 3/3 Post-Processing Data...")

post_processing.visualize(options, analysed_conversations)

print("✅ Finished Post-Processing Data")