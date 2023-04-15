import pandas as pd
from io import open as io_open
import sys
import os
from datetime import datetime
import argparse
import numpy as np
import random
import math
from operator import itemgetter

from collections import OrderedDict



def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
#    print(lines[0])
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

def write2DListToLineFile(file_path, output_list, shard, out_dir):
    line_fns = []
    for idx, line in enumerate(output_list):

        cur_line = "line" + str(idx)
        line_file = file_path + "-" + cur_line + ".txt"

        # we concatenate the from 2:
        line_fns.append(line_file)
        writeListFile(line_file, [line])
    shard_text_fn_file = "shard" + str(shard) + "_fn_file.txt"
    shard_text_fn_file_path = os.path.join(out_dir, shard_text_fn_file)
    writeListFile(shard_text_fn_file_path, line_fns)

def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

def write2DListFile(file_path, output_list, line_sep = " "):
    str_list = []
    for out_line in output_list:
        str_line = []
        for e in out_line:
            str_line.append(str(e))
        str_list.append(str_line)
    out_list = list(map(line_sep.join, str_list))
    writeListFile(file_path, out_list)

def writeSetFile(file_path, output_set, sort = True):
    output_list = list(output_set)
    if sort:
        output_list.sort()
    writeListFile(file_path, output_list)

def readNEMSISFile(file_path, discard_header = True, encoding = None):
    lines = readFile(file_path, encoding)
    if discard_header:
        lines = lines[1:]
    no_join_lines = []
    for line in lines:
        t = line.strip().split('~|~')
        t = [e.strip() for e in t]
        no_join_lines.append(t)
    return no_join_lines

def getMultiValDictFromFile(file_path, sep = ' ', encoding = None):
    # some keys may have multiple values, so
    # we concatenate all the values for the unique key
    lines = readFile(file_path, encoding)
    lines = [s.split(sep) for s in lines]
    d = dict()
    same_val_count = 0
    diff_val_count = 0
    for line in lines:
        k = line[0].strip()
        v = line[1].strip()
        if k in d and v not in d[k]:
            cur_val = d[k] + sep + v
            d[k] = cur_val
            diff_val_count += 1
        elif k in d and v in d[k]:
            same_val_count += 1
        elif k not in d:
            d[k] = v
    print("same_val_count: %s" % same_val_count)
    print("diff_val_count: %s" % diff_val_count)
    print("dict_size: %s" % len(d))
    print("total_lines: %s" % len(lines))
    assert(same_val_count + diff_val_count + len(d) == len(lines))
    return d

def convertMultiValCode2Dscp(code_file_path, ref_file_path, out_file_path):

    d = getDictFromFile(ref_file_path, sep = '~|~')
    lines = readFile(code_file_path)
    lines = [s.split('~|~') for s in lines]
#    new_d = dict()
    new_d = OrderedDict()
    unmatch_code = 0
    match_code = 0
    total_code = 0
    for line in lines:
        k = line[0]
        vals = line[1:]
        dscp_val = ""
        for val in vals:
            total_code += 1
            if val in d:
                dscp_val = dscp_val + " " +d[val]
                match_code += 1
            else:
                unmatch_code += 1
        if dscp_val:
            new_d[k] = dscp_val
    writeDictToFile(out_file_path, new_d)
    print("total lines: %s, total new lines: %s, total codes: %s, match code count: %s, unmatch code count: %s" % (len(lines), len(new_d), total_code, match_code, unmatch_code))
 
def getDictFromFile(file_path, sep = ' ', encoding = None):
    lines = readFile(file_path, encoding)
    lines = [s.split(sep) for s in lines]
    d = dict()
    for line in lines:
        k = line[0].strip()
        v = line[1].strip()
        if k in d:
            print("same item key")
        d[k] = v
    return d

def writeDictToFile(file_path, d, sep = '~|~', encoding = None):

    out_list = []
    for k, v in d.items():
        out_line = str(k) + sep + str(v)
        out_list.append(out_line)
    writeListFile(file_path, out_list, encoding = encoding)

def read2DArrayFromFile(file_path, line_sep = ' ', dtype = int):
    lines = readFile(file_path)
    arr = []
    for row in lines:
        cols = row.split()
        arr_cols = []
        for col in cols:
            arr_cols.append(dtype(col))
        arr.append(arr_cols)
    return arr
