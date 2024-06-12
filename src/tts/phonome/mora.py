from src.tts.kana.param import (
    SMALL_REGEX, 
    YOUON_WHITE_LIST, 
    WHITE_LIST,
    kana2phonome
)

from src.tts.phonome.base import (
    extra_symbols,
    ACCENT_UP
)

from src.tts.kana.utils import remove_choon

_extra_symbols = [ACCENT_UP, *extra_symbols]

def split_mora(kana):
    """かな文字をモーラごとに分割

    Args:
        kana (_type_): ^コ[レワ_キャ]ナシイモノ#ガタリダッタ$

    Returns:
        _type_: ['^', 'コ', '[', 'レ', 'ワ', '_', 'キャ', ']', 'ナ', 'シ', 'イ', 'モ', 'ノ', '#', 'ガ', 'タ', 'リ', 'ダ', 'ッ', 'タ', '$']
    """
    mora_list = []
    for i, char in enumerate(kana): 
        if SMALL_REGEX.match(char):
            # 特定の小文字の場合モーラとしては登録しない
            # 一つ前のモーラと結合して拗音のホワイトリストに含まれている形式になるかを確認する
            # ならなければエラーとする
            youon_str = mora_list[-1] + char
            if not youon_str in YOUON_WHITE_LIST:
                raise ValueError(f"{kana}の{youon_str}は中間言語で使用できない文字です。")
            
            # 直前のモーラを修正
            mora_list[-1] = youon_str
        
        else:
            mora_list.append(char)
    
    for m in mora_list:
        # 記号や特定の小文字以外の場合はモーラとして登録して良い文字か判定した後登録する
        if not (m in WHITE_LIST or m in YOUON_WHITE_LIST or m in _extra_symbols):
            raise ValueError(f"{kana}の{m}は中間言語で使用できない文字です。")
        
    return mora_list

def kana_to_symbols(kana):
    """韻律付きかな文字をシンボル（音素）に変換する

    Args:
        kana (_type_):^コ[レワ_キャ]ナシイモノ#ガタリダッタ$

    Returns:
        _type_: ['^', 'k', 'o', '[', 'r', 'e', 'w', 'a', '_', 'ky', 'a', ']', 'n', 'a', 'sh', 'i', 'i', 'm', 'o', 'n', 'o', '#', 'g', 'a', 't', 'a', 'r', 'i', 'd', 'a', 'cl', 't', 'a', '$']
    """
    kana = remove_choon(kana)
    kana_list = split_mora(kana)
    symbol_list = []
    for char in kana_list:
        if char in kana2phonome:
            char = kana2phonome[char].split(" ")
            symbol_list.extend(char)
        elif char in _extra_symbols:
            symbol_list.append(char)
        else:
            raise ValueError(f"{char}に関して定義されていません。")
        
    return symbol_list

if __name__ == "__main__":
    text = "^コ[レワ_キャ]ナシイモノ#ガタリダッタ$"
    print(split_mora(text))
    print(kana_to_symbols(text))