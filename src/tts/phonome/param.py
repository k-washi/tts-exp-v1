from src.tts.phonome.base import (
    SOS,
    EOS,
    QUESTION_MARK,
    POSE,
    ACCENT_BOUNDARY,
    ACCENT_UP,
    ACCENT_DOWN,
    MORA_END_LIST,
    phonomes,
    _pad,
    _phonome_map,
)

phonomes = phonomes + [QUESTION_MARK, POSE, ACCENT_BOUNDARY, SOS, EOS] + [_pad] + [ACCENT_UP, ACCENT_DOWN]
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

def symbol_preprocess(
    symbol_list, 
    add_blank_type=0, 
    accent_split=False,
    accent_up_ignore=False
):
    """
    Preprocesses a list of symbols by mapping phonemes, splitting accents, and adding blanks.

    Args:
        symbol_list (list): The list of symbols to be preprocessed.
        add_blank_type (int, optional): The type of blank to add. Defaults to 0.
            0: Do not add blanks.
            1: Add a blank after phoneme.
            2: Add a blank after each mora.
        accent_split (bool, optional): Whether to split accents. Defaults to False.
        accent_up_ignore (bool, optional): Whether to ignore accent_up. Defaults to False.

    Returns:
        tuple: A tuple containing the preprocessed phoneme list and accent list.

    Raises:
        ValueError: If an invalid symbol or add_blank_type value is encountered.
        AssertionError: If the lengths of the phoneme list and accent list do not match.
    """
    
    phonome_list, accent_list = [], []
    for p in symbol_list:
        if p in _remove_symbol_list:
            continue
        if p in _phonome_map:
            p = _phonome_map[p]
        
        if accent_up_ignore and p == ACCENT_UP:
            # アクセントの上昇を無視
            continue
        
        if accent_split:
            # アクセントを分割する
            if p in phonomes and p not in accents:
                # アクセントでない場合、音素リストに追加
                phonome_list.append(p)
                accent_list.append(_pad)
            
            if p in accents:
                # アクセントの場合、アクセントの音素リストの最後をアクセントに入れ替える
                if phonome_list[-1] not in [_pad]:
                    accent_list[-1] = p
                elif len(phonome_list) > 1 and phonome_list[-2] not in [_pad]:
                    accent_list[-2] = p
                else:
                    raise ValueError(f"アクセントの前に音素がありません。{phonome_list}")
        else:
            # アクセントを分割しない
            phonome_list.append(p)
            accent_list.append(_pad)
        if p in accents:
            # アクセントのときは無視
            pass
        elif add_blank_type == 0:
            # ブランクを追加しない
            pass
        elif add_blank_type == 1:
            # 1つ空白を追加
            phonome_list.append(_pad)
            accent_list.append(_pad)
        elif add_blank_type == 2:
            # モーラごとに空白を追加
            if p in MORA_END_LIST:
                phonome_list.append(_pad)
                accent_list.append(_pad)
        else:
            raise ValueError(f"add_blank_typeの値が不正です。{add_blank_type}")
        
        if p not in phonomes and p not in accents:
            raise ValueError(f"{p}は音素リストに含まれていません。")
    
    assert len(phonome_list) == len(accent_list), f"音素リストとアクセントリストの長さが一致しません。"
    return phonome_list, accent_list

if __name__ == "__main__":
    test_list = [
        "^-m-i-[-z-u-o-#-m-a-[-r-e-]-e-sh-i-a-k-a-r-a-#-k-a-[-w-a-n-a-]-k-u-t-e-w-a-#-n-a-[-r-a-]-n-a-i-n-o-d-e-s-u-$"
    ]
    
    for text in test_list:
        print("==="*10)
        text = text.split("-")
        print(symbol_preprocess(text, add_blank_type=0, accent_split=False))
        print(symbol_preprocess(text, add_blank_type=0, accent_split=True))
        print(symbol_preprocess(text, add_blank_type=1, accent_split=True))
        print(symbol_preprocess(text, add_blank_type=2, accent_split=True))
        print(symbol_preprocess(text, add_blank_type=2, accent_split=True, accent_up_ignore=True))