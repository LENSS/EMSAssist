import pandas as pd
from io import open as io_open
import sys
import os
from datetime import datetime
import argparse
import numpy as np
import yaml
# import matplotlib.pyplot as plt
import random
import math
import re

from collections import OrderedDict


def readFile(file_path, encoding = None):
  res = []
  with open(file_path, 'r') as f:
    for idx, line in enumerate(f):
      line = line.strip()
      res.append(line)
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
    print("write %s lines to %s" % (len(output_list), file_path))

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

def whole_word_found(str, word):
    if re.search(r"\b" + re.escape(str) + r"\b", word):
        return True
    return False

if __name__ == "__main__":   
    
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description = "utils to preprocess nemsis")

    t_total = datetime.now() - t_start
    print("this run takes %s" % t_total)
