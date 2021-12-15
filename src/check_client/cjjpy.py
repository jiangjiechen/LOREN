# -*- coding: utf-8 -*-

'''
@Author : Jiangjie Chen
@Time   : 2018/11/15 17:08
@Contact: jjchen19@fudan.edu.cn
'''

import re
import datetime
import os
import argparse
import logging
import traceback

try:
    import ujson as json
except:
    import json

HADOOP_BIN = 'PATH=/usr/bin/:$PATH hdfs'
FOR_PUBLIC = True


def LengthStats(filename):
    len_list = []
    thresholds = [0.8, 0.9, 0.95, 0.99, 0.999]
    with open(filename) as f:
        for line in f:
            len_list.append(len(line.strip().split()))
    stats = {
        'Max': max(len_list),
        'Min': min(len_list),
        'Avg': round(sum(len_list) / len(len_list), 4),
    }
    len_list.sort()
    for t in thresholds:
        stats[f"Top-{t}"] = len_list[int(len(len_list) * t)]

    for k in stats:
        print(f"- {k}: {stats[k]}")
    return stats


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def TraceBack(error_msg):
    exc = traceback.format_exc()
    msg = f'[Error]: {error_msg}.\n[Traceback]: {exc}'
    return msg


def Now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def AbsParentDir(file, parent='..', postfix=None):
    ppath = os.path.abspath(file)
    parent_level = parent.count('.')
    while parent_level > 0:
        ppath = os.path.dirname(ppath)
        parent_level -= 1
    if postfix is not None:
        return os.path.join(ppath, postfix)
    else:
        return ppath


def init_logger(log_file=None, log_file_level=logging.NOTSET, from_scratch=False):
    from coloredlogs import ColoredFormatter
    import tensorflow as tf

    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    log_format = ColoredFormatter(fmt=fmt)
    # log_format = logging.Formatter()
    logger = logging.getLogger()
    logger.setLevel(log_file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if from_scratch and tf.io.gfile.exists(log_file):
            logger.warning('Removing previous log file: %s' % log_file)
            tf.io.gfile.remove(log_file)
        path = os.path.dirname(log_file)
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def OverWriteCjjPy(root='.'):
    # import difflib
    # diff = difflib.HtmlDiff()
    cnt = 0
    golden_cjjpy = os.path.join(root, 'cjjpy.py')
    # golden_content = open(golden_cjjpy).readlines()
    for dir, folder, file in os.walk(root):
        for f in file:
            if f == 'cjjpy.py':
                cjjpy = '%s/%s' % (dir, f)
                # content = open(cjjpy).readlines()
                # d = diff.make_file(golden_content, content)
                cnt += 1
                print('[%d]: %s' % (cnt, cjjpy))
                os.system('cp %s %s' % (golden_cjjpy, cjjpy))


def ChangeFileFormat(filename, new_fmt):
    assert type(filename) is str and type(new_fmt) is str
    spt = filename.split('.')
    if len(spt) == 0:
        return filename
    else:
        return filename.replace('.' + spt[-1], new_fmt)


def CountLines(fname):
    with open(fname, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1  # Remove this if a wc-like count is needed
    return count


def GetDate():
    return str(datetime.datetime.now())[5:10].replace('-', '')


def TimeClock(seconds):
    sec = int(seconds)
    hour = int(sec / 3600)
    min = int((sec - hour * 3600) / 60)
    ssec = float(seconds) - hour * 3600 - min * 60
    # return '%dh %dm %.2fs' % (hour, min, ssec)
    return '{0:>2d}h{1:>3d}m{2:>6.2f}s'.format(hour, min, ssec)


def StripAll(text):
    return text.strip().replace('\t', '').replace('\n', '').replace(' ', '')


def GetBracket(text, bracket, en_br=False):
    # input should be aa(bb)cc, True for bracket, False for text
    if bracket:
        try:
            return re.findall('\（(.*?)\）', text.strip())[-1]
        except:
            return ''
    else:
        if en_br:
            text = re.sub('\(.*?\)', '', text.strip())
        return re.sub('（.*?）', '', text.strip())


def CharLang(uchar, lang):
    assert lang.lower() in ['en', 'cn', 'zh']
    if lang.lower() in ['cn', 'zh']:
        if uchar >= '\u4e00' and uchar <= '\u9fa5':
            return True
        else:
            return False
    elif lang.lower() == 'en':
        if (uchar <= 'Z' and uchar >= 'A') or (uchar <= 'z' and uchar >= 'a'):
            return True
        else:
            return False
    else:
        raise NotImplementedError


def WordLang(word, lang):
    for i in word.strip():
        if i.isspace(): continue
        if not CharLang(i, lang):
            return False
    return True


def SortDict(_dict, reverse=True):
    assert type(_dict) is dict
    return sorted(_dict.items(), key=lambda d: d[1], reverse=reverse)


def lark(content='test'):
    print(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--diff', nargs=2,
                        help='show difference between two files, shown in downloads/diff.html')
    parser.add_argument('--de_unicode', action='store_true', default=False,
                        help='remove unicode characters')
    parser.add_argument('--link_entity', action='store_true', default=False,
                        help='')
    parser.add_argument('--max_comm_len', action='store_true', default=False,
                        help='')
    parser.add_argument('--search', nargs=2,
                        help='search key from file, 2 args: file name & key')
    parser.add_argument('--email', nargs=2,
                        help='sending emails, 2 args: subject & content')
    parser.add_argument('--overwrite', action='store_true', default=None,
                        help='overwrite all cjjpy under given *dir* based on *dir*/cjjpy.py')
    parser.add_argument('--replace', nargs=3,
                        help='replace char, 3 args: file name & replaced char & replacer char')
    parser.add_argument('--lark', nargs=1)
    parser.add_argument('--get_hdfs', nargs=2,
                        help='easy copy from hdfs to local fs, 2 args: remote_file/dir & local_dir')
    parser.add_argument('--put_hdfs', nargs=2,
                        help='easy put from local fs to hdfs, 2 args: local_file/dir & remote_dir')
    parser.add_argument('--length_stats', nargs=1,
                        help='simple token lengths distribution of a line-by-line file')

    args = parser.parse_args()

    if args.overwrite:
        print('* Overwriting cjjpy...')
        OverWriteCjjPy()

    if args.lark:
        try:
            content = args.lark[0]
        except:
            content = 'running complete'
        print(f'* Larking "{content}"...')
        lark(content)

    if args.length_stats:
        file = args.length_stats[0]
        print(f'* Working on {file} lengths statistics...')
        LengthStats(file)
