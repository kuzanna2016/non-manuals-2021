import re


def find_name(video_name):
    name = re.search(r'[-_]([a-zA-Z]+)$', video_name).groups('')[0].lower()
    if name == 'g':
        name = 'gul'
    elif name == 'mir':
        name = 'mira'
    return name


def find_sentence(video_name):
    name = re.search(r'^[_a-zA-Z.]+[-_][a-zA-Z.]+[-_]([a-zA-Z._]+)[-_][a-zA-Z.]+$', video_name).groups('')[0]
    if '.' in name:
        name = name.replace('.', '_')
        name = name.replace('postroili', 'post')
        name = name.replace('ustala', 'ust')
        name = name.replace('bezhit', 'beg')
        name = name.replace('sobaka', 'sob')
        name = name.replace('uch', 'uchit')
        name = name.replace('uc_smeh', 'uchit_smeh')
        name = name.replace('mam_ust', 'mama_ust')
        name = name.replace('pap_beg', 'papa_beg')
        name = name.replace('mail', 'mal')
        name = name.replace('razbilos', 'razb')
    elif 'devu' in name:
        name = 'dev_upala'
    elif 'mam_ust' == name:
        name = 'mama_ust'
    elif name == 'pap_beg':
        name = 'papa_beg'
    elif name == 'uc_smeh':
        name = 'uchit_smeh'
    return name


def find_type(video_name):
    name = re.search(r'(^[a-z]+[_-](?:q[_-])?[a-z1]+)[_-]', video_name).groups('')[0]
    name = name.replace('-', '_')
    name = name.replace('1', '')
    name = name.replace('utv', 'st')
    if name == 'kar_n':
        name = 'st_neut'
    elif name == 'st_amger':
        name = 'st_anger'
    elif name == 'gen_q':
        name = 'gen_q_neut'
    return name


def change_type(name):
    name = name.replace('-', '_')
    name = name.replace('question', 'q')
    name = name.replace('general', 'gen')
    name = name.replace('_neutral', '')
    name = name.replace('_neut', '')
    name = name.replace('partial', 'part')
    name = name.replace('statement', 'st')
    return name


def proper_name(video_name):
    name = find_name(video_name)
    sentence = find_sentence(video_name)
    sentence_type = change_type(find_type(video_name))
    return f'{sentence_type}-{sentence}-{name}'
