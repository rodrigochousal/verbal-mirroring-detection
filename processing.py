def extract_analyses(options, conversations):
    analysed_conversations = []
    for c in conversations:
        for analysis, is_on in options.requested_analyses.items():
            if not is_on: continue
            print(f"Performing {analysis} analysis of recordings...")
            if analysis == 'p2r':
                p2r_conversation = prompt_to_response(c)
                analysed_conversations.append(p2r_conversation)
            if analysis == 'r2r':
                r2r_conversation = response_to_response(c)
                analysed_conversations.append(r2r_conversation)
    return analysed_conversations

def prompt_to_response(conversation):
    '''
    For each speaker in a conversation, calculates a feature's ratio of prompt to response.
    E.g. response is 'i'th utterance and prompt is 'i-1'th utterance, each from a different speaker.

    Args:
    conversation: A Conversation object containing a 2D matrix of utterances.

    Returns:
    p2r_conversation: A Conversation object containing prompt:repsonse ratio value
    '''
    # Calculate average feature ratio of prompt:response for each utterance
    p2r_conversation = conversation
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
        p2r_conversation.utterances[i].p2r = prev_nz_value/u.value
    return p2r_conversation

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
    r2r_conversation = conversation
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
        r2r_conversation.utterances[i].r2r = speaker_change/prompter_change
    return r2r_conversation