import unicodedata
import string

ALL_CHARACTERS = string.ascii_letters + " .,;'" + "_"


def unicode_to_ascii(sentence: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_CHARACTERS
    )


