"""Explicit coarse classes for an external zero-shot Urdu timing aligner.

These classes CANNOT verify native vowel length, nasality, aspiration or
dental/retroflex distinctions. They only locate durations for independently
admitted Urdu targets; target symbols/IDs remain unchanged. Calibration with
wrong-transcript controls is required before training admission.
"""
import numpy as np

GROUPS={
    'a':('a','aː','ɑ','ɑː','ə','ɐ','ʌ','ɜ','ɜː'),
    'i':('i','iː','ɪ','ᵻ','ĩ','ĩ'),
    'u':('u','uː','ʊ','ũ','ũ'),
    'e':('e','eː','e̞','ɛ','ɛː','æː','ẽ','ẽː','ɛ̃'),
    'o':('o','oː','o̞','ɔ','ɔː','õ','ɔ̃'),
    't':('t','t̪','ʈ','tʰ','ʈʰ'), 'd':('d','ɖ','dʰ','ɖʰ'),
    'p':('p','pʰ','ph'), 'b':('b','bʰ'),
    'k':('k','kʰ','kh','q'), 'ɡ':('ɡ','ɡʰ','ɣ'),
    'tʃ':('tʃ','tʃʰ'), 'dʒ':('dʒ',),
    'r':('r','ɾ','ɹ','ɽ'), 'n':('n','ɳ','ɲ'),
    'l':('l','ɭ'), 'v':('v','ʋ','w'), 'h':('h','ħ','ʔ'),
}
PROXIES={member:primary for primary,members in GROUPS.items() for member in members}
PROXIES.update({'t̪ʰ':'t','d̪':'d','d̪ʱ':'d','bʱ':'b','ɡʱ':'ɡ','ɖʱ':'d','dʒʱ':'dʒ',
                'ɦ':'h','ɽʱ':'r','mʱ':'m','nʱ':'n','lʱ':'l','rʱ':'r',
                'ɑ̃ː':'a','ə̃':'a','ɪ̃':'i','ĩː':'i','ʊ̃':'u','ũː':'u',
                'ẽː':'e','ɛ̃ː':'e','õː':'o','ɔ̃ː':'o'})


def project(log_probs,vocabulary):
    """Move probability mass into disjoint classes; do not inflate its sum."""
    result=log_probs.copy()
    used=set()
    for primary,members in GROUPS.items():
        if primary not in vocabulary:raise ValueError(f'missing aligner class {primary}')
        ids=[vocabulary[s] for s in members if s in vocabulary]
        if used.intersection(ids):raise ValueError('overlapping acoustic classes')
        used.update(ids)
        result[:,ids]=-np.inf
        result[:,vocabulary[primary]]=np.logaddexp.reduce(log_probs[:,ids],axis=1)
    return result
