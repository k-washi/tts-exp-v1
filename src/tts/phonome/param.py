from .base import (
    SOS,
    EOS,
    QUESTION_MARK,
    POSE,
    ACCENT_BOUNDARY,
    ACCENT_UP,
    ACCENT_DOWN,
    phonomes,
    _pad,
    _phonome_map
)

phonomes = phonomes + [QUESTION_MARK, POSE, ACCENT_BOUNDARY, SOS, EOS] + [_pad]
accents = [ACCENT_UP, ACCENT_DOWN] + [_pad]
phonomes_to_id = {p: i for i, p in enumerate(phonomes)}
id_to_phonomes = {i: p for i, p in enumerate(phonomes)}
accents_to_id = {a: i for i, a in enumerate(accents)}
id_to_accents = {i: a for i, a in enumerate(accents)}

_remove_symbol_list = [
    #SOS,
    #EOS
]

def num_phonomes():
    return len(phonomes)

def num_accents():
    return len(accents)

def phonome_to_sequence(phonome_list):
    return [phonomes_to_id[p] for p in phonome_list]

def sequence_to_phonome(sequence_list):
    return [id_to_phonomes[i] for i in sequence_list]

def accent_to_sequence(accent_list):
    return [accents_to_id[a] for a in accent_list]

def sequence_to_accent(sequence_list):
    return [id_to_accents[i] for i in sequence_list]

def symbol_preprocess(symbol_list, add_blank=False):
    """前処理

    Args:
        symbol_list (_type_): 音素などのリスト

    Returns:
        _type_: _description_
    """
    phonome_list, accent_list = [], []
    for p in symbol_list:
        if p in _remove_symbol_list:
            continue
        if p in _phonome_map:
            p = _phonome_map[p]
        
        if p in phonomes:
            phonome_list.append(p)
            accent_list.append(_pad)
            
        if p in accents:
            # アクセントの場合、アクセントの音素リストの最後をアクセントに入れ替える
            accent_list[-1] = p
        
        if add_blank:
            # 1つ空白を追加
            phonome_list.append(_pad)
            accent_list.append(_pad)
        
        if p not in phonomes and p not in accents:
            raise ValueError(f"{p}は音素リストに含まれていません。")
        
    return phonome_list, accent_list