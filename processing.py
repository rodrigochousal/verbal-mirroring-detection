def prompt_to_response(conversation):
    '''
    For each speaker in a conversation, calculates a feature's ratio of prompt to response.
    E.g. response is 'i'th utterance and prompt is 'i-1'th utterance, each from a different speaker.

    Args:
    conversation: A Conversation object containing a 2D matrix of utterances.

    Returns:
    speaker_p2r_ratios: A dictionary containing speaker_id:[(prompt:repsonse ratio)]
    '''
    # Calculate average feature ratio of prompt:response for each utterance
    speaker_p2r_ratios = {}
    # For each utterance
    for i, u in enumerate(conversation.utterances):
        # Ignore silence or first utterance
        if (u.speaker_id == -1) or (i == 0): continue
        # Find previous non-zero utterance
        prev_nz_value = -1
        for j in range(i-1, 0, -1):
            jth_value = conversation.utterances[j].value
            if (jth_value > 0):
                prev_nz_value = jth_value
                break
        if (prev_nz_value == -1): continue
        # Calculate average feature ratio for prompt:response non-zero utterance
        sid = u.speaker_id
        p2r = prev_nz_value/u.value
        if sid in speaker_p2r_ratios:
            speaker_p2r_ratios[sid].append((p2r, u.length))
        else:
            speaker_p2r_ratios[sid] = [(p2r, u.length)]
    return speaker_p2r_ratios

def response_to_response(conversation):
    '''
    For each speaker in a conversation, calculates a feature's average ratio of response to response.
    E.g. response is 'i'th utterance and response is 'i-1'th utterance, s.t. each utterance is from the same speaker.

    Args:
    conversation: A Conversation object containing a 2D matrix of utterances.

    Returns:
    None
    '''
    # Calculate relative change in average feature value for same speaker's responses
    # compared to the prompter's relative change
    speaker_r2r_ratios = {}
    conversation_rr_ratios = []
    # For each significant utterance
    for i, u in enumerate(conversation.utterances):
        # Ignore first and second utterances
        if (i < 3): continue
        # Find same speaker's previous utterance
        sid = u.speaker_id
        curr_value_0 = u.value
        prev_value_0 = -1 # this could be u.prev.value
        for j in range(i-1, 0, -1):
            jth_value = conversation.utterances[j].value
            jth_id = conversation.utterances[j].speaker_id
            if (jth_id == sid):
                prev_value_0 = jth_value
                break
        if (prev_value_0 == -1.0): continue
        # Find different speaker's previous utterance
        prompt_utterance = conversation.utterances[i-1]
        sid = prompt_utterance.speaker_id
        curr_value_1 = prompt_utterance.value
        prev_value_1 = -1 # this could be u.prev.value
        for j in range(i-2, 0, -1):
            jth_value = conversation.utterances[j].value
            jth_id = conversation.utterances[j].speaker_id
            if (jth_id == sid):
                prev_value_1 = jth_value
                break
        if (prev_value_1 == -1.0): continue
        # Calculate r2r
        speaker_change = curr_value_0/prev_value_0
        prompter_change = curr_value_1/prev_value_1
        r2r = speaker_change/prompter_change
        if sid in speaker_r2r_ratios:
            speaker_r2r_ratios[sid].append((r2r, u.length))
        else:
            speaker_r2r_ratios[sid] = [(r2r, u.length)]
    for key, ratios in speaker_r2r_ratios.items():
        sum_of_ratios = 0
        for ratio in ratios:
            sum_of_ratios += ratio[0]
        average_ratio = sum_of_ratios/len(ratios)
        print(f"Speaker {key} average R2R ratio: {average_ratio:.4f}")