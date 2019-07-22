import os
import sys
import importlib
import re
import editdistance

SPACE_NORMALIZER = re.compile(r"\s+")


def remove_tone_line(utf8_str):
    utf8_str = utf8_str.lower()
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = [ch for ch in str(intab_l + intab_u)]

    outtab_l = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d"
    outtab_u = "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"
    outtab = outtab_l + outtab_u

    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    # add char without tone and lowercase
    label_chars = line.split()[1:]
    label_chars_simply = [remove_tone_line(char_item) for char_item in label_chars]
    label_chars += label_chars_simply
    return label_chars


def acc_pair_string(str_1, str_2, is_word_level=True):
    if is_word_level:
        str_1 = str_1.split()
        str_2 = str_2.split()

    return (1. - editdistance.distance(str_1, str_2) / max(len(str_1), len(str_2))) * 100


def import_user_module(module_path):
    if module_path is not None:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


if __name__ == '__main__':
    output_str = [
        'toi di choi',
        'troi dep qua'
    ]

    ground_truth_str = [
        'toi di choi',
        'troi dep qua'
    ]

    cer = 0
    wer = 0
    for index, (o_item, g_item) in enumerate(zip(output_str, ground_truth_str)):
        wer += (acc_pair_string(o_item, g_item, is_word_level=True) - wer) / (index + 1)
        cer += (acc_pair_string(o_item, g_item, is_word_level=False) - cer) / (index + 1)
    print('cer: ', cer)
    print('wer: ', wer)
