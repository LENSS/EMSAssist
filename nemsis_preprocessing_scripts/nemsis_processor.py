import sys
import pandas as pd
import re
import csv
import yaml
import Utils as util
import os
import natsort
import codecs
from collections import OrderedDict
from datetime import datetime
import argparse

pri_sym_ref_name = "ESITUATION_09REF.txt"
pri_imp_ref_name = "ESITUATION_11REF.txt"
add_sym_ref_name = "ESITUATION_10REF.txt"
sec_imp_ref_name = "ESITUATION_12REF.txt"

fact_pcr_ps_file_name = "FACTPCRPRIMARYSYMPTOM.txt"
fact_pcr_pi_file_name = "FACTPCRPRIMARYIMPRESSION.txt"
fact_pcr_as_file_name = "FACTPCRADDITIONALSYMPTOM.txt"
fact_pcr_si_file_name = "FACTPCRSECONDARYIMPRESSION.txt"
cached_pcr2si_file_name = "nemsis_pcr2si.txt"

class NEMSIS_Processor(object):
    
  def __init__(self, nemsis_dir, nemsis_year, cache_dir):

    pri_sym_ref = os.path.join(nemsis_dir, nemsis_year, pri_sym_ref_name)
    pri_imp_ref = os.path.join(nemsis_dir, nemsis_year, pri_imp_ref_name)
    add_sym_ref = os.path.join(nemsis_dir, nemsis_year, add_sym_ref_name)
    sec_imp_ref = os.path.join(nemsis_dir, nemsis_year, sec_imp_ref_name)

    self.ref_files = [pri_sym_ref, pri_imp_ref, add_sym_ref, sec_imp_ref]
    self.d_list, self.global_d, self.code_map, self.word_map = self.get_dict()
    print("RefInfo: d_list length %s, global_d length %s, code_map length %s, word_map length %s" % 
          (len(self.d_list), len(self.global_d), len(self.code_map), len(self.word_map)))

    self.cached_pcr2si_file_path = os.path.join(cache_dir, nemsis_year, cached_pcr2si_file_name)

  def get_dict(self):

    d_list = []
    global_d = dict()

    for ref_idx, ref_file_path in enumerate(self.ref_files):
      # print("RefInfo file path %s" % (ref_file_path))
      ref_f_lines = util.readFile(ref_file_path)
      ref_f_lines = [s.split("~|~") for s in ref_f_lines]
      d = dict()

      for i, line in enumerate(ref_f_lines):

        # print("line_num = %s, %s" % (i, line))

        if i == 0:
          continue

        k = line[0].strip()
        v = line[1].strip()

        if (k.lower() == "unknown") or (v.lower() == "unknown"):
          continue

        v = v.lower()

        if k in d:
          assert d[k] == v
        d[k] = v

        if k in global_d:
          if global_d[k] != v:
            print(global_d[k], v)
          assert global_d[k] == v
        global_d[k] = v

      d_list.append(d)

    codes_list = list(global_d.keys())
    codes_list = natsort.natsorted(codes_list)
#    writeListFile("sorted_codes_list.txt", codes_list)
    codes_map = dict()
    for (i, code) in enumerate(codes_list):
      codes_map[code] = i
  
    words_set = set()
    for v in global_d.values():
      words_set.update(v.split())
    words_list = list(words_set)
    words_list = natsort.natsorted(words_list)
#    writeListFile("sorted_words_list.txt", words_list)
    words_map = dict()
    for (i, word) in enumerate(words_list):
      words_map[word] = i
  
    return d_list, global_d, codes_map, words_map

  def is_valid_si(self, si_code):
    return (si_code in self.global_d)

  def get_valid_pcr_set_from_si_file(self, file_path, prev_pcr_set = None):

    curr_pcr_set = set()
    if prev_pcr_set == None:
      # print("prev_pcr_set is None")
      with open(file_path, "r") as r_f:
        for idx, line in enumerate(r_f):
          event = line.strip().split('~|~')
          assert len(event) == 2

          pcr_k, si_code = event[0].strip(), event[1].strip()
          # narrow down the pcr_set
          if self.is_valid_si(si_code):
            curr_pcr_set.add(pcr_k)
    else:
      # print("prev_pcr_set is not None")
      with open(file_path, "r") as r_f:
        for idx, line in enumerate(r_f):
          event = line.strip().split('~|~')
          assert len(event) == 2

          pcr_k, si_code = event[0].strip(), event[1].strip()
          # narrow down the pcr_set
          if self.is_valid_si(si_code) and pcr_k in prev_pcr_set:
            curr_pcr_set.add(pcr_k)

    print("obtaining valid pcr from %s, curr_pcr_set size %s" % (file_path, len(curr_pcr_set)))

    return curr_pcr_set

  def get_pcr2si_from_cached_file(self, cached_file_path):

    pcr2si = dict()
    with open(cached_file_path, "r") as r_f:
      for idx, line in enumerate(r_f):
        event = line.strip().split('~|~')
        assert len(event) == 5
        
        pcr_k = event[0].strip()
        si = "~|~".join(event[1:])
        pcr2si[pcr_k] = si
    
    print("obtaining valid pcr2si from cached file %s, cached pcr2si size %s" % 
          (cached_file_path, len(pcr2si)))
    
    return pcr2si

  def get_pcr2si(self,
                 nemsis_dir,
                 nemsis_year,
                 cached_pcr2si_file_path):

    # cached_file_path = self.cached_pcr2si_file_path
    if os.path.isfile(cached_pcr2si_file_path):
      return self.get_pcr2si_from_cached_file(cached_pcr2si_file_path)

    ps_file_path = os.path.join(nemsis_dir, nemsis_year, fact_pcr_ps_file_name)
    pi_file_path = os.path.join(nemsis_dir, nemsis_year, fact_pcr_pi_file_name)
    as_file_path = os.path.join(nemsis_dir, nemsis_year, fact_pcr_as_file_name)
    si_file_path = os.path.join(nemsis_dir, nemsis_year, fact_pcr_si_file_name)

    ps_pcr_set = self.get_valid_pcr_set_from_si_file(ps_file_path)
    ps_pi_pcr_set = self.get_valid_pcr_set_from_si_file(pi_file_path, ps_pcr_set)
    ps_pi_as_pcr_set = self.get_valid_pcr_set_from_si_file(as_file_path, ps_pi_pcr_set)
    ps_pi_as_si_pcr_set = self.get_valid_pcr_set_from_si_file(si_file_path, ps_pi_as_pcr_set)

    # signs and symptoms can be duplicate
    pcr2si = dict()
    print("ps_pi_as_si_pcr_set length ", len(ps_pi_as_si_pcr_set))
    pcr2si = self.augment_pcr_with_si(ps_pi_as_si_pcr_set, ps_file_path, pcr2si)
    pcr2si = self.augment_pcr_with_si(ps_pi_as_si_pcr_set, pi_file_path, pcr2si)
    pcr2si = self.augment_pcr_with_si(ps_pi_as_si_pcr_set, as_file_path, pcr2si)
    pcr2si = self.augment_pcr_with_si(ps_pi_as_si_pcr_set, si_file_path, pcr2si)

    with open(cached_pcr2si_file_path, "w") as w_f:
      for pcr, si in pcr2si.items():
        line = pcr + si + "\n"
        w_f.write(line)

    print("write %s lines to %s" % (len(pcr2si), cached_pcr2si_file_path))

    return self.get_pcr2si_from_cached_file(cached_pcr2si_file_path)
  
  def augment_pcr_with_si(self, pcr_set, input_si_file_path, pcr2si):

    pcr_accessed = set()
    with open(input_si_file_path, "r") as r_f:
      for row, line in enumerate(r_f):
        if row == 0:
          continue

        event = line.strip().split('~|~')
        assert len(event) == 2
        pcr_k = event[0].strip()
        si_code = event[1].strip()

        if (pcr_k in pcr_set) and (si_code in self.global_d):
          if (si_code not in self.global_d):
            print("[row %s]: %s not in self.global_d in file %s" % (row, si_code, input_si_file_path))
          # assert si_code in self.global_d

          if pcr_k not in pcr_accessed:    # have not appended current si before
            if pcr_k not in pcr2si:        # have not appended ps before
              pcr2si[pcr_k] = "~|~" + si_code
            else:
              pcr2si[pcr_k] += "~|~" + si_code
          else:
            # print("multiple si exist for one pcr")
            pcr2si[pcr_k] += " " + si_code
          pcr_accessed.add(pcr_k)

    print("augmented pcr2si size %s" % len(pcr2si))
    return pcr2si

def write_pcr2si_file(nemsis_dir, nemsis_year, cache_dir):

  np = NEMSIS_Processor(nemsis_dir, nemsis_year, cache_dir)

  # if "Unknown" in np.global_d:
  #   print("global_d has an Unkown key", np.global_d["Unknown"])
  # else:
  #   print("global_d has no empty key")
  # exit()

  pcr2si = np.get_pcr2si(nemsis_dir, nemsis_year, cache_dir)

if __name__ == "__main__":

  t1 = datetime.now()

  parser = argparse.ArgumentParser(description = "control the functions for EMSBrain")
  parser.add_argument("--nemsis_dir", action='store', type=str, default = "/slot1/NEMSIS_Databases")
  parser.add_argument("--nemsis_year", action='store', type=str, default = "2020")
  parser.add_argument("--cache_dir", action='store', type=str, default = "nemsis_cache_files")
  args = parser.parse_args()

  write_pcr2si_file(args.nemsis_dir, args.nemsis_year, args.cache_dir)
  # p1 = NEMSIS_Processor(args.nemsis_dir, args.nemsis_year)

  t2 = datetime.now()
  print("This run takes:", t2 - t1)