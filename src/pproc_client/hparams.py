# -*- coding: utf-8 -*-

"""
@Author     : Jiangjie Chen
@Time       : 2020/12/10 14:07
@Contact    : jjchen19@fudan.edu.cn
@Description:
"""

import os


QG_PREFIX = '%s/data/fact_checking/{version}/cache_qg/' % os.environ['PJ_HOME']
AG_PREFIX = '%s/data/fact_checking/{version}/cache_ag/' % os.environ['PJ_HOME']

CACHED_QUESTION_FILE = 'question.{role}.cache'
CACHED_ANSEWR_FILE = 'answer.{role}.cache'
CACHED_EVIDENTIAL_FILE = 'evidential.k_{k_cand}.{role}.cache'
FINAL_FILE = '{role}.json'