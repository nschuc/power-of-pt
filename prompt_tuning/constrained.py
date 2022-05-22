from typing import List
from functools import partial


_end = object()

def make_trie(sentences: List[List[int]], end=_end):
    """Builds a trie from sentences list"""
    root = dict()
    for sent in sentences:
        curr = root
        for tok in sent:
            curr = curr.setdefault(tok, {})
        curr[end] = end
        
    return root

def valid_next_tokens(trie, prefix: List[int]):
    """Returns valid next token continuations given prefix"""
    curr = trie

    for tok in prefix:
        curr = curr.get(tok)
        if not curr:
            return []
    
    return [ c for c in curr.keys() if c != _end ]

def is_in_trie(trie, eos, seq):
    curr = trie

    for tok in seq:
        if tok == eos:
            return True

        curr = curr.get(tok)
        if not curr:
            return False
    
    if tok != eos:
        return False

    return True

def build_prefix_allowed_tokens_fn(valid_targets, eos):
    trie = make_trie(valid_targets, end=eos)

    def allowed_tokens(batch_id, input_ids):
        input_ids = input_ids.tolist()
        valid_tokens = valid_next_tokens(trie, input_ids[1:])
        return valid_tokens

    return allowed_tokens, partial(is_in_trie, trie, eos)