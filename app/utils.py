import re


def tokenize(text: str):
    return text.lower().split()


def clean_and_format_text(raw_text: str) -> str:
    text = raw_text.replace("\\n", "\n")

    text = re.sub(r"(?<=\n)(\d+\.|\*|-) +", r"\1 ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"(?<!\n)\* ", r"\n* ", text)
    return text.strip()


def join_strings_in_dict(input_dict):
    return {key: ' '.join(value) for key, value in input_dict.items()}
