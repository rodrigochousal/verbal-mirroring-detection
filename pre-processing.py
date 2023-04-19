def downsample(data, window_size):
    """
    Downsample data by taking the mean of consecutive groups of 'window_size' elements.
    """
    downsampled = []
    for i in range(0, len(data), window_size):
        chunk = data[i:i+window_size]
        if chunk: downsampled.append(sum(chunk) / len(chunk))
    return downsampled

def make_utterances(data):
    length = len(data)
    downsampled_data = downsample(data, length/100)
    