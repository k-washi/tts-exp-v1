import pyopenjtalk
from pathlib import Path

from src.tts.frontend.openjtalk import pp_symbols
from src.tts.phonome.base import symbols
from src.tts.phonome.mora import kana_to_symbols

def save_symbols(phonome_list, output_fp, join_mark="-"):
    t = f"{join_mark}".join(phonome_list)
    with open(output_fp, "w") as f:
        f.write(t)
        
def read_symbols(fp, join_mark="-"):
    with open(fp, "r") as f:
        t = f.read().replace("\n", "")
        t = t.split(join_mark)
    return t

def extract_symbols(text):
    """日本語の文章から音素+アクセントを抽出する

    Args:
        text (str): 日本語文

    Returns:
        List[str]: 音素+アクセントのリスト
    """
    symbols = pp_symbols(pyopenjtalk.extract_fullcontext(text))
    return symbols

class OpenJtalk():
    def __init__(self, user_dic=None) -> None:
        if user_dic is not None and Path(user_dic).is_file():
            pyopenjtalk.set_user_dict(user_dic)
    
    def extract_symbols(self, text):
        # 漢字かな交じり文字から、音素+アクセントに変換
        return extract_symbols(text)
    
    def kana2symbols(self, text):
        return kana_to_symbols(text)