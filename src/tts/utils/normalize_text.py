from __future__ import unicode_literals

import re
import regex
import unicodedata

phone_regex_text = "\d{1,6}[-ー－]\d{1,4}[-ー－]\d{1,6}"
yuubin_regex_text = "\d{3,3}[-ー－]\d{4,4}"
ataib_regex_text = "\d{1,3}[-ー－]\d{1,3}"

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    """
    余計なスペースは除去
    """
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    """
    全角の記号は半角に変換、大文字は小文字に
    """
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def regex_process(text):
    # 現在、入力前に"-"を"－"に変換している
    phone_re = regex.compile(phone_regex_text)
    yuubin_re = regex.compile(yuubin_regex_text)
    ataib_re = regex.compile(ataib_regex_text)
    
    # 電話番号に関する処理
    phone_iter = list(phone_re.finditer(text))
    for obj in phone_iter:
        rep_text = text[obj.start(0):obj.end(0)].replace("－", "-").replace("ー", "-")
        text = text[:obj.start(0)] + rep_text + text[obj.end(0):]
    
    # 郵便番号に関する処理
    yuubin_iter = list(yuubin_re.finditer(text))
    for obj in yuubin_iter:
        rep_text = text[obj.start(0):obj.end(0)].replace("－", "-").replace("ー", "-")
        text = text[:obj.start(0)] + rep_text + text[obj.end(0):]
        
    # A対Bに関する処理
    ataib_iter = list(ataib_re.finditer(text))
    for obj in ataib_iter:
        rep_text = text[obj.start(0):obj.end(0)].replace("－", "対").replace("ー", "対")
        text = text[:obj.start(0)] + rep_text + text[obj.end(0):]
       
    return text

def post_process(text):
    haihun_char = "－"
    text = text.replace("-", haihun_char)
    text = regex_process(text)
    return text
    
def normalize_text(text):
    """
    textの前処理
    """
    text = text.replace("\n", "").replace("\r", "").replace("\t", " ")
    text = normalize_neologd(text)
    text = post_process(text)

    return text

if __name__ == "__main__":
    text_list = [
        [
            "COVID－19の影響で旅行は中止になりました。",
            "COVID－19の影響で旅行は中止になりました。"
        ],
        [
            "10-３でした。",
            "10対3でした。"
        ],
        [
            "090－1234-5678",
            "090-1234-5678"
        ],
        [
            "090－1234",
            "090-1234"
        ]
    ]
    
    for input_text, correct_label in text_list:
        print(normalize_text(input_text))
        print(correct_label)
        print(normalize_text(input_text) == correct_label)
        print("=====================================")