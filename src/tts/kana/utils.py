import src.tts.kana.param as kana_param

def remove_choon(text):
    """
    Removes the choon (long vowel) character 'ー' from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the choon characters removed.
    """
    mora_list = []
    result_list = []
    # 一旦モーラ単位に分割する(モーラとして認められるかのチェックは行わず記号も一モーラとする)
    for text_char in text:  
        if kana_param.SMALL_REGEX.match(text_char):
            youon_str = mora_list[-1] + text_char
            mora_list[-1] = youon_str
        else:
            mora_list.append(text_char)
    
    for mora_char in mora_list:
        if mora_char == "ー":
            # 長音の場合一つ前のモーラを取得し、その母音をモーラとして登録する。
            # 一つ前のモーラが母音の辞書に登録されていなければもう一つ前を参照する。(「ズッート」なら「ズ」「イッチョ!ー」なら「チョ」の母音とする)
            for reverse_char in reversed(result_list):
                replaced_char = kana_param.BOIN_DICT.get(reverse_char, None)
                if replaced_char is not None:
                    result_list.append(replaced_char)
                    break
        else:
            result_list.append(mora_char)

    return "".join(result_list)

if __name__ == "__main__":
    # text = "ゴ^ヒャクメ!ートル"
    text = '[イ!ッチョー]'
    import regex
    print(remove_choon(text))

    text = '[イ!ッチョー]エンヲモッテ[ヒャクメートル]ハシルサトー'
    print("#1:", remove_choon(text))
    result_text = ""
    cup_regex = regex.compile("\[.+?\]+")
    cup_iter = cup_regex.finditer(text)
    for in_cup in cup_iter:
        regex_result_text = in_cup.group()
        regex_text_start = in_cup.start(0)
        regex_text_end = in_cup.end(0)
        print(text)
        removed_part = remove_choon(regex_result_text)
        print(removed_part)
        print(regex_text_start)
        print(regex_text_end)
        print(text[:regex_text_start])
        print(text[regex_text_end:])
        result_text += text[:regex_text_start] + removed_part + text[regex_text_end:]
    print(result_text)