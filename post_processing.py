import numpy as np
import matplotlib.pyplot as plt

def visualize(options, conversations):
    for c in conversations:
        for analysis, is_on in options.requested_analyses.items():
            if not is_on: continue
            print(f"Performing {analysis} analysis of recordings...")
            if analysis == 'p2r':
                plot_p2r(c)
            if analysis == 'r2r':
                plot_r2r(c)

def plot_p2r(conversation):
    '''
    Plot a scatter plot with values y being float and x being time.

    Args:
    speaker_ratios: A dictionary containing speaker_id:[ratio]

    Returns:
    None
    '''
    for id in conversation.unique_speaker_ids():
        time_values = []
        analysis_values = []
        for u in conversation.utterances:
            if u.speaker_id == id:
                if u.p2r is not None:
                    analysis_values.append(u.p2r)
                    time_values.append(u.start_time)
        time_values = np.array(time_values)
        analysis_values = np.array(analysis_values)
        if len(time_values) == 0 or len(analysis_values) == 0:
            # skip if no valid values found
            continue
        try:
            # find line of best fit
            a, b = np.polyfit(time_values, analysis_values, 1)
        except TypeError as e:
            # print error message and skip if polyfit() fails
            print(f"Error: {e}")
            continue
        # add points to plot
        plt.scatter(time_values, analysis_values, label=f"Speaker {id}", s=20)
        # add line of best fit to plot
        plt.plot(time_values, a*time_values+b)
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
    conversation: A Conversation object

    Returns:
    None
    '''
    # Extract time-relevant data
    for id in conversation.unique_speaker_ids():
        time_values = []
        analysis_values = []
        for u in conversation.utterances:
            if u.speaker_id == id:
                if u.r2r is not None:
                    analysis_values.append(u.r2r)
                    time_values.append(u.start_time)
        time_values = np.array(time_values)
        analysis_values = np.array(analysis_values)
        if len(time_values) == 0 or len(analysis_values) == 0:
            # skip if no valid values found
            continue
        try:
            # find line of best fit
            a, b = np.polyfit(time_values, analysis_values, 1)
        except TypeError as e:
            # print error message and skip if polyfit() fails
            print(f"Error: {e}")
            continue
        # add points to plot
        plt.scatter(time_values, analysis_values, label=f"Speaker {id}", s=20)
        # add line of best fit to plot
        plt.plot(time_values, a*time_values+b)

    # set the x-axis label, y-axis label, and title
    plt.xlabel('Time (s)')
    plt.ylabel('R2R Ratio')
    plt.title('Speaker Ratios')
    plt.legend()

    # show the plot
    plt.show()
