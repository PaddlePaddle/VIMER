"""imgtool function
"""
import numpy as np

def truncate_seq_pair(tokens_b, max_length):
    """truncate_seq_pair
    """
    while True:
        total_length = len(tokens_b)
        if total_length <= max_length:
            break
        tokens_b.pop()
def process_sent(caption, tokenizer, max_seq_length=20):
    """process_sent
    """
    tokens_caption = tokenizer.tokenize(caption)
    truncate_seq_pair(tokens_caption, max_seq_length - 2)
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_caption:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*(len(input_ids))
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return [np.array(input_ids), np.array(input_mask), np.array(segment_ids)]
