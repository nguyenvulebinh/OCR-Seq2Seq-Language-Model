from fairseq_cli import train, preprocess
import json
import random
import visen
import re
import traceback
import sys
from tqdm import tqdm
from multiprocessing import Pool

json_data = [
    '/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/data-bin/address/1.json',
    '/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/data-bin/address/2.json',
    '/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/data-bin/address/3.json',
    '/Users/nguyenbinh/Programing/Python/OCR-Seq2Seq-Language-Model/data-bin/address/4.json'
]
valid_rate = 0.1
num_process = 5
trainpref = "./data-bin/address/train"
validpref = "./data-bin/address/valid"


def get_raw_address_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_data:
        address_json = json.loads(file_data.read())
    return list(address_json.values())


def remove_space(input_str):
    return ''.join(i if i != ' ' or random.randint(0, input_str.count(' ') * 2) else '' for i in input_str)


def is_word(word, mark_char='_'):
    special_char = list('0123456789')
    special_char.append(mark_char)
    for item in special_char:
        if item in word:
            return False
    return True


def edit_char(input_str):
    def edits1(word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        list_choose = list(set(deletes + transposes + replaces + inserts))
        if len(list_choose) > 0:
            return random.choice(list_choose)
        else:
            return word

    def edits2(word):
        return random.choice([word.lower(), word.upper()])

    def misspeller(word):
        option = [edits1, edits2, edits1]
        return random.choice(option)(word)

    misspelled_s = ''
    misspelled_list = []
    for item in input_str.split(' '):
        if is_word(item):
            if random.randint(0, len(input_str)):
                misspelled_list.append(item)
            else:
                misspelled_list.append(misspeller(item))
        else:
            misspelled_list.append(item)
    misspelled_s = ' '.join(misspelled_list)
    misspelled_s = re.sub(r'\s+', ' ', misspelled_s)
    return misspelled_s


def remove_char(input_str, mark_char='_'):
    return ''.join(i if (i == mark_char or random.randint(0, len(input_str))) else '' for i in input_str)


def write_file(file_pref, data):
    with open(file_pref + ".correct", 'w', encoding='utf-8') as file_correct:
        with open(file_pref + ".error", 'w', encoding='utf-8') as file_error:
            for sen_correct, sen_error in data:
                file_correct.write(' '.join(list(sen_correct.replace(' ', '|'))) + "\n")
                file_error.write(' '.join(list(sen_error.replace(' ', '|'))) + "\n")


def make_spell_error(input_str):
    raw_input_str = input_str
    try:
        # hide info is not word
        # input_str, list_hidden_info = hide_info(input_str)

        option = [remove_space, edit_char, remove_char]
        list_func = [visen.remove_tone]
        for i in range(len(option)):
            list_func.append(random.choice(option))
        output_str = input_str
        for func in list_func:
            output_str = func(output_str)
        # restore info is not word
        # output_str = restore_hidden_info(output_str, list_hidden_info)
        return raw_input_str, output_str.lower()
    except:
        print("input error: {}\n".format(raw_input_str), traceback.format_exc())
        return raw_input_str


def clean_input_correction(raw_input):
    input_str = raw_input.lower()
    input_str = visen.remove_tone(input_str)
    input_str = ' '.join(list(input_str.replace(' ', '|')))
    return input_str


def clean_output_correction(raw_output):
    output_str = ''.join(list(raw_output.replace(' ', ''))).replace('|', ' ')
    return output_str


if __name__ == '__main__':
    list_address = []
    for item in json_data:
        list_address += get_raw_address_from_json(item)
    list_address = list(set(list_address))
    random.shuffle(list_address)
    valid_correct = list_address[:int(len(list_address) * valid_rate)]
    train_correct = list_address[int(len(list_address) * valid_rate):]

    pool = Pool(num_process)
    valid = list(
        tqdm(pool.imap_unordered(make_spell_error, valid_correct), total=len(valid_correct), desc="Make valid data"))
    train = list(
        tqdm(pool.imap_unordered(make_spell_error, train_correct), total=len(train_correct), desc="Make train data"))
    pool.close()

    write_file(trainpref, train)
    write_file(validpref, valid)

    sys.argv += [
        "--source-lang", "error",
        "--target-lang", "correct",
        "--trainpref", trainpref,
        "--validpref", validpref,
        "--destdir", "./data-bin/address/processed",
        "--workers", "2"
    ]
    preprocess.cli_main()
