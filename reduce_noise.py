# perform noise reduction
from scipy.io import wavfile
import noisereduce as nr

# load data
for i in range(1,7):
    for j in range (1,4):
        filepath = "./data/audio/conv_0" + str(i) + "_pid" + str (j) + ".wav"
        rate, data = wavfile.read(filepath)
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        clean_filepath = "./data/audio/conv_0" + str(i) + "_pid" + str (j) + "_clean.wav"
        wavfile.write(clean_filepath, rate, reduced_noise)