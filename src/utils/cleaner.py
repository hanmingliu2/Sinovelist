import logging
import re

import pandas as pd
from mapping import (
    FRENCH_MAP,
    GERMAN_MAP,
    GREEK_MAP,
    PUNCTUATION_MAP,
    RUSSIAN_MAP,
    SPANISH_MAP,
    WHITESPACES,
)
from path import METADATA_CSV, NOVELS_FOLDER

# Log INFO level messages in console
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def capitalize_english_letters(text: str) -> str:
    """
    Capitalize all english letters in the text.

    In Chinese novels, there are often common English abbreviations like SUV, ICU, etc..
    Capitalizing them reduces vocabulary size for a character-level LLM.
    """
    return text.upper()


def clean_extra_whitespaces(text: str) -> str:
    """
    Remove extra whitespaces around punctuations.

    Please  call this function before calling `clean_punctuations`,
    as this function expects Chinese punctuations.
    """

    # Remove all whitespaces except "\n"
    spaces_to_remove = WHITESPACES.copy()
    spaces_to_remove.remove("\n")
    spaces_to_remove = "".join(list(spaces_to_remove))

    for punctuation in PUNCTUATION_MAP.keys():
        segments = text.split(punctuation)
        segments = [s.strip(spaces_to_remove) for s in segments]
        text = punctuation.join(segments)

    # In Chinese, double quotes in consecutive spoken sentences remain unambiguous.
    #   “你好”“早上好”“你这是去上班了？”
    # That's not no longer the case with English equivalents.
    #   "你好""早上好""你这是去上班了?"
    # Add a space between consecutive double quotes to prevent confusions,
    # before replacing them with English equivalents.
    text = text.replace("”“", "” “")
    return text


def clean_chinese_punctuations(text: str) -> str:
    """
    1. Repalce Chinese punctuations with English equivalents.
    2. Replace full-width punctuations with half-width equivalents.

    E.g.
    顿号: 、-> ,
    逗号: ，-> ,
    句号: 。-> .
    感叹号: ！-> !
    etc.

    A Chinese punctuation typically takes up 3 bytes in UTF-8,
    whereas an English punctuation takes up only 1 byte.
    This replacement can reduce text size significantly.
    """
    table = str.maketrans(PUNCTUATION_MAP)
    text = text.translate(table)

    # Replace Chinese ellipses with ...
    text = text.replace("……", "...").replace("…", "...")
    # Drop unecessary escape character (\) in plain text
    text = text.replace("\\", "")
    return text


def clean_website_clutters(text: str) -> str:
    """
    Remove editorial words, website branding, and disclaimers.

    Chinese novelists often have editorial words at the end of a chapter,
    in a much more casual tone, which is inconsistent with the novel's writing style.
    Removing them helps an LLM learns better. Website branding and disclaimers also removed,
    as they are unrelated to the novel.

    Please call this function:
       1. after calling `clean_punctuations`
       2. before calling `clean_shapter_headers`,
    as this function relies on:
       1. English punctuations
       2. capther headers for pattern matching.
    """

    old_lines = text.splitlines()
    new_lines = []
    skip = False
    for line in old_lines:
        # Website branding
        if "WWW.52SHUKU.VIP" in line:
            skip = True
            continue

        # Disclaimer
        if line.startswith("附:本作品来自互联网,本人不做任何负责,内容版权归作者所有!"):
            skip = True
            continue

        # Editorial words often begins with "作者有话", and ends before the next chapter starts.
        if re.match(r"^作者有话[要想]*说", line):
            skip = True
            continue

        if skip:
            patterns = [
                r"第\d+[章卷]",
                r"第\d+[章卷]\s*\(.*\)",
                r"第\d+[章卷]\s*番外.*",
                r"第[零一二三四五六七八九十百千]+[章卷]",
                r"第[零一二三四五六七八九十百千]+[章卷]\s*\(.*\)",
                r"第[零一二三四五六七八九十百千]+[章卷]\s*番外.*",
                r"[IVXLCDM]+\.",
            ]
            # Reached start of next chapter, reset flag
            if any(re.fullmatch(pattern, line) for pattern in patterns):
                skip = False

        if not skip:
            new_lines.append(line)
    return "\n".join(new_lines).strip("\n")


def clean_chapter_headers(text: str) -> str:
    """
    Remove lines that are chapter headers."

    Patterns
    ----------
    1. Arabic numerals, may have extra words after
        E.g. 第1章, 第2章 (已修), 第100章 番外
    2. Similar to 1. but with Chinese numerals
        E.g. 第一章, 第二章 (已修), 第一百章 番外
    3. Roman numerals, usually end with a period
        E.g. I. II. IV. X.
    """

    patterns = [
        r"第\d+[章卷]",
        r"第\d+[章卷]\s*\(.*\)",
        r"第\d+[章卷]\s*番外.*",
        r"第[零一二三四五六七八九十百千]+[章卷]",
        r"第[零一二三四五六七八九十百千]+[章卷]\s*\(.*\)",
        r"第[零一二三四五六七八九十百千]+[章卷]\s*番外.*",
        r"[IVXLCDM]+\.",
    ]
    old_lines = text.splitlines()
    new_lines = []
    for line in old_lines:
        if any(re.fullmatch(pattern, line) for pattern in patterns):
            new_lines.append("")
        else:
            new_lines.append(line)
    return "\n".join(new_lines).strip("\n")


def clean_footers(text: str) -> str:
    """
    Remove footers that marks the end of a novel.

    Chinese novelists often puts markers like "-全文完-" or "__END__" at the end of a novel.
    It's random as in everyone writes them differently.
    Removing them reduces noises in the training data.
    """

    patterns = [
        r"[_-]*(?:THE )?END[_-]*",
        r"[_-]*全文完(?:结)?[_-]*",
    ]
    old_lines = text.splitlines()
    new_lines = []
    for line in old_lines:
        if any(re.fullmatch(pattern, line) for pattern in patterns):
            new_lines.append("")
        else:
            new_lines.append(line)
    return "\n".join(new_lines).strip("\n")


def clean_pinyin(text: str) -> str:
    """
    Replace Pinyin with the corresponding Chinese character in the text.

    Pinyin is a romanization system for Mandarin Chinese.
    It uses Latin alphabets to spell Chinese characters phonetically.

    Certain Chinese characters are considered "sensitive" due to censorship.
    So novelists use Pinyin to spell them out to get around censorship.
    Raplce them with the actual characters can save quite a bit of memory.
    E.g. replacing CHUÁNG with 床 saves 4 bytes.
    """

    pinyin_map = {
        "BÀO": "暴",
        "BĪ": "逼",
        "CHUÁNG": "床",
        "CHUĪ": "吹",
        "CHŪN": "春",
        "DÀNG": "荡",
        "DÒNG": "洞",
        "GĀN": "干",
        "HUÁNG": "黄",
        "JIĀO": "交",
        "JĪ": "鸡",
        "JĪNG": "精",
        "LUǑ": "裸",
        "LÀNG": "浪",
        "LÁNG": "狼",
        "PÀO": "炮",
        "QIÁNG": "强",
        "RǓ": "乳",
        "XUÉ": "血",
        "YĪN": "阴",
    }

    for pinyin, character in pinyin_map.items():
        text = text.replace(pinyin, character)
    return text


def clean_symbols(text: str) -> str:
    """
    1. Remove rare symbols that are likely typos.
    2. Remove name separator '·'.
    3. Replace symbols with proper Chinese equivalents.
    4. Replace cuss words with (口吐芬芳) (a sarcastic way to say cussing in Chinese).

    Reasons
    ----------
    1. Removing rare symbols reduces vocab size and reduces noise.
    2. The '·' are misused in many places; dropping it improve syntax correctness.
    3. Replacing symbols with Chinese equivalent reduces vocab size.
        ° -> 度 (degree)
        ℃ -> 摄氏度 (Celsius)
        ℉ -> 华氏度 (Fahrenheit)
        § -> 章节 (section)
        etc.
    4. Chinese novelists often use random symbols to describe someone cussing angrily:
        "走路不看#@￥..."司机的叫骂渐渐远去
        "@#￥%#!..."贡阿驰用方言低声呵斥了几句
       Replacing it with "(口吐芬芳)" reduces random patterns the LLM has to learn.
    """

    # Drop these weird symbols
    text = re.sub(r"[□☆]", "", text)

    # Drop the name separator
    text = text.replace("·", "")

    # Replace symbols with Chinese equivalents, may expand the mapping in the future.
    symbol_map = {
        "°": "度",
        "℃": "摄氏度",
        "℉": "华氏度",
        "§": "章节",
        "×": "X",
        "÷": "/",
        "艹": "草",  # Equivalent to the F-word in English
    }
    for symbol, equivalent in symbol_map.items():
        text = text.replace(symbol, equivalent)

    # Replace cuss words patterns with a placeholder
    text = re.sub(r"[!#@%&\*\$\(\)]{2,}\.{3}", "(口吐芬芳)", text)
    return text


def clean_language(text: str) -> str:
    """
    Replace characters in other languages with English equivalents.

    Chinese novels, just like other novels, come in many different genres and topics.
    Some contains words in other languages like Spanish, French, German, etc.
    Replacing those characters into English equivalents reduces vocabulary size.
    """

    # Ordering matters here as some languages have over lapping characters.
    # For simplicity, I will ignore this for now.
    for m in [GREEK_MAP, SPANISH_MAP, FRENCH_MAP, GERMAN_MAP, RUSSIAN_MAP]:
        translation_table = str.maketrans(m)
        text = text.translate(translation_table)

    # Capitalize it again
    return capitalize_english_letters(text)


def main():
    df = pd.read_csv(METADATA_CSV)
    df["filename"] = df["author_name"] + "_" + df["novel_name"] + ".txt"
    df["filepath"] = df["filename"].apply(lambda f: NOVELS_FOLDER / f)
    df["is_scraped"] = df["filepath"].apply(lambda p: p.exists())

    # Check if there is any scraped novel
    df = df[df["is_scraped"]]
    if df.empty:
        LOGGER.error("Cleaning failed: couldn't find any scraped novel")
        return

    for filepath in df["filepath"]:
        with open(filepath, "r", encoding="utf-8") as f1:
            text = f1.read()
            text = capitalize_english_letters(text)
            text = clean_extra_whitespaces(text)
            text = clean_chinese_punctuations(text)
            text = clean_website_clutters(text)
            text = clean_chapter_headers(text)
            text = clean_footers(text)
            text = clean_pinyin(text)
            text = clean_symbols(text)
            text = clean_language(text)

            filename, suffix = str(filepath).split(".")
            with open(f"{filename}_cleaned.{suffix}", "w", encoding="utf-8") as f2:
                f2.write(text)


if __name__ == "__main__":
    main()
