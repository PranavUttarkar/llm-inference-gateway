"""Factor controls and randomized execution order; no answer labels are used."""
import random

CONDITIONS = {
    'solo': {'batch': 1, 'extra_padding': 0, 'position': 0, 'composition': 'short'},
    'solo_repeat': {'batch': 1, 'extra_padding': 0, 'position': 0, 'composition': 'short'},
    'solo_padded': {'batch': 1, 'extra_padding': 128, 'position': 0, 'composition': 'short'},
    'batch8': {'batch': 8, 'extra_padding': 0, 'position': 0, 'composition': 'short'},
    'batch8_repeat': {'batch': 8, 'extra_padding': 0, 'position': 0, 'composition': 'short'},
    'batch8_content': {'batch': 8, 'extra_padding': 0, 'position': 0, 'composition': 'alternate'},
    'batch8_padded': {'batch': 8, 'extra_padding': 128, 'position': 0, 'composition': 'short'},
    'batch8_last': {'batch': 8, 'extra_padding': 0, 'position': 7, 'composition': 'short'},
}


def schedule(question_ids, seed=1701):
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError('need nonempty unique question IDs')
    jobs = [{'question_id': key, 'condition': condition} for key in question_ids for condition in CONDITIONS]
    random.Random(seed).shuffle(jobs)
    return jobs


def padded_batch(target, condition, pad_id, companion_tokens):
    """Right padding keeps target tokens/positions identical; explicit masks follow.

    Companions are constructed once per question and shared across geometry arms.
    Composition controls retain identical length/mask; no target truncation occurs.
    """
    settings = CONDITIONS[condition]
    length = len(target) + settings['extra_padding']
    if not target or len(companion_tokens) != 7:
        raise ValueError('need target tokens and seven companions')
    sequences = [list(tokens) for tokens in companion_tokens]
    if any(not tokens or len(tokens) > len(target) for tokens in sequences):
        raise ValueError('companions must be nonempty and no longer than target')
    if settings['composition'] == 'alternate':
        sequences = [list(reversed(tokens)) for tokens in sequences]
    if settings['batch'] == 1:
        sequences = [list(target)]
    else:
        sequences.insert(settings['position'], list(target))
    ids = [tokens + [pad_id] * (length - len(tokens)) for tokens in sequences]
    masks = [[1] * len(tokens) + [0] * (length - len(tokens)) for tokens in sequences]
    return ids, masks, settings['position'], len(target) - 1
