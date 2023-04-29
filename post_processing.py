import matplotlib.pyplot as plt
import numpy as np

def plot_p2r(conversation):
    '''
    Plot a scatter plot with values y being float and x being time.

    Args:
    speaker_ratios: A dictionary containing speaker_id:[ratio]

    Returns:
    None
    '''
    # Extract time-relevant data
    for id in conversation.unique_speaker_ids():
        time_values = []
        analysis_values = []
        for u in conversation.utterances:
            if u.speaker_id == id:
                time_values.append(u.start_time)
                analysis_values.append(u.p2r)
        plt.plot(time_values, analysis_values, label=id)

    # set the x-axis label, y-axis label, and title
    plt.xlabel('Time (s)')
    plt.ylabel('P2R Ratio')
    plt.title('Speaker Ratios')
    plt.legend()

    # show the plot
    plt.show()

def plot_r2r(conversation):
    '''
    Plot a scatter plot with values y being float and x being time.

    Args:
    speaker_ratios: A dictionary containing speaker_id:[ratio]

    Returns:
    None
    '''
    # Extract time-relevant data
    for id in conversation.unique_speaker_ids():
        time_values = []
        analysis_values = []
        for u in conversation.utterances:
            if u.speaker_id == id:
                time_values.append(u.start_time)
                analysis_values.append(u.r2r)
        plt.plot(time_values, analysis_values, label=id)

    # set the x-axis label, y-axis label, and title
    plt.xlabel('Time (s)')
    plt.ylabel('R2R Ratio')
    plt.title('Speaker Ratios')
    plt.legend()

    # show the plot
    plt.show()