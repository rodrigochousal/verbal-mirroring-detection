import sys
import os

# list of valid audio file extensions
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac']

# get command line arguments (excluding the first argument which is the name of the script)
file_paths = sys.argv[1:]

# iterate through each file path and check if it is an audio file
for path in file_paths:
    # get the file extension
    ext = os.path.splitext(path)[1].lower()

    # check if the file extension is in the list of valid audio extensions
    if ext not in AUDIO_EXTENSIONS:
        print(f"{path} is not an audio file.")
        raise SystemExit(1)

