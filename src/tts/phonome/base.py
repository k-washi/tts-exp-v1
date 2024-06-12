# 音声合成に使用するシンボルを定義
SOS = "^"
EOS = "$"
QUESTION_MARK = "?"
POSE = "_"
ACCENT_BOUNDARY = "#"
ACCENT_UP = "["
ACCENT_DOWN = "]"

extra_symbols = [
    SOS,  # 文の先頭を表す特殊記号 <SOS>
    EOS,  # 文の末尾を表す特殊記号 <EOS> (通常)
    QUESTION_MARK,  # 文の末尾を表す特殊記号 <EOS> (疑問系)
    POSE,  # ポーズ
    ACCENT_BOUNDARY,  # アクセント句境界
    ACCENT_UP,  # ピッチの上がり位置
    ACCENT_DOWN,  # ピッチの下がり位置
]

MORA_END_LIST = ["a", "i", "u", "e", "o", "cl", "N"]

phonomes = [
    'a',
    'i',
    'u',
    'e',
    'o',
    'k',
    's',
    't',
    'n',
    'h',
    'm',
    'y',
    'r',
    'w',
    'g',
    'z',
    'd',
    'p',
    'b',
    'ky',
    'gy',
    'sh',
    'j',
    'ch',
    'ny',
    'dy',
    'f',
    'hy',
    'py',
    'by',
    'v',
    'my',
    'ry',
    'cl',
    'ty',
    'N',
    'ts',
]

_pad = "~"

symbols = [_pad] + extra_symbols + phonomes
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_phonome_map = {
}

_remove_symbol_list = [
    #"["
]