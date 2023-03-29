from pydub import AudioSegment

audio_path = "./data/audio/clean/conv_0{}_pid{}_clean.wav"
output_path = "conv_0{}_overlay.wav"
for i in range(1,7):
    # Load the three audio files
    audio_p1 = AudioSegment.from_wav(audio_path.format(i,1))
    audio_p2 = AudioSegment.from_wav(audio_path.format(i,2))
    audio_p3 = AudioSegment.from_wav(audio_path.format(i,3))

    # Set the length of the output file to the length of the input files
    length = len(audio_p1)

    # Overlay the three audio files into a single file
    output = audio_p1.overlay(audio_p2).overlay(audio_p3)

    # Save the output file
    output.export(output_path.format(i), format="wav")