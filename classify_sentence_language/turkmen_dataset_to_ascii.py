import re

from utils import unicode_to_ascii


def process_turkmen_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    sentences = re.split(r'(?<=[.!?])\s+', content)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for sentence in sentences:
            cleaned_sentence = re.sub(r'\d{1,2}:\d{2} \(.*?\)', '', sentence).replace('\n', ' ').strip()
            ascii_sentence = unicode_to_ascii(cleaned_sentence)

            if ascii_sentence and len(ascii_sentence.split()) > 3:
                outfile.write(ascii_sentence + '\n')


# any Turkmen dataset that contains non ascii characters like 'ä', 'ö', 'ü', etc.
input_file = 'data/test.txt'
output_file = 'data/test2.txt'
process_turkmen_dataset(input_file, output_file)
