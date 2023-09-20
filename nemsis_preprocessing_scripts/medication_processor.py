import collections
import argparse
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from datetime import datetime
import Utils as util
import tokenization
import optimization
import os
import csv
import random
from absl import logging
from collections import Counter, defaultdict
import tempfile
logging.set_verbosity(logging.INFO)
import quantization_configs as quant_configs

from nemsis_processor import NEMSIS_Processor as NP

# global_seed = 1993
# pub_pcr_events_file_name = "Pub_PCRevents.txt"

fact_pcr_medication_file_name = "FACTPCRMEDICATION.txt"
eMedications_01_date_time_index = 0
PcrMedicationKey_foreign_k_index = 1
PcrKey_pcr_k_index = 2
eMedications_03_medication_given_index = 3
eMedications_03Descr_index = 4
eMedications_05_medication_dosage_index = 5
eMedications_06_medication_dosage_units_index = 6
eMedications_07_response_to_medication_index = 7
eMedications_10_role_of_person_administering_medication_index = 8
eMedications_02_medication_administering_before_EMS_index = 9

nemsis_RxNorm2Med_file_name = "RxNorm2Medication.csv"
cached_si2med_file_name = "nemsis_si2med.txt"
cached_si2med_med_set_file_name = "nemsis_si2med_med_set.txt"
cached_med2desc_file_name = "nemsis_med2desc.txt"
cached_desc2med_file_name = "nemsis_desc2med.txt"

class Medication_Processor(object):

  def __init__(self, nemsis_dir, nemsis_year, cache_dir):

    self.nemsis_dir = nemsis_dir
    self.nemsis_year = nemsis_year
    self.cache_dir = cache_dir
    # self.code2med_dict, self.med2code_dict = self.read_code2med()
    self.cached_si2med_file_path = os.path.join(cache_dir, nemsis_year, cached_si2med_file_name)

    self.med_set_file_path = os.path.join(cache_dir, nemsis_year, cached_si2med_med_set_file_name)
    self.cached_med2desc_file_path = os.path.join(cache_dir, nemsis_year, cached_med2desc_file_name)
    self.cached_desc2med_file_path = os.path.join(cache_dir, nemsis_year, cached_desc2med_file_name)

  def read_cached_si2med(self, file_path):

    si2med = dict()
    with open(file_path, "r") as r_f:

      for row, line in enumerate(r_f):
        if row == 0:
          continue

        event = line.strip().split('~||~')
        si = event[0].strip()
        med_line = event[1].strip()
        si2med[si] = med_line

    print("retrieved %s si2med from %s with %s lines of cached si2med, the key set size = %s, the value set size = %s" % 
          (len(si2med), file_path, row+1, len(si2med.keys()), len(si2med.values())))
    return si2med

  def get_si2med(self):

    if os.path.isfile(self.cached_si2med_file_path):
      return self.read_cached_si2med(self.cached_si2med_file_path)

    nemsis_med_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_medication_file_name)
    np = NP(self.nemsis_dir, self.nemsis_year, self.cache_dir)
    nemsis_pcr2si = np.get_pcr2si(self.nemsis_dir, self.nemsis_year, np.cached_pcr2si_file_path)

    # the date index is move to the end of the column before nemsis 2020
    pcr_k_index = PcrKey_pcr_k_index if int(self.nemsis_year) >= 2020 else PcrKey_pcr_k_index - 1
    med_index = eMedications_03_medication_given_index if int(self.nemsis_year) >= 2020 else eMedications_03_medication_given_index - 1
    med_desc_index = eMedications_03Descr_index if int(self.nemsis_year) >= 2020 else eMedications_03Descr_index - 1
    med_improve_index = eMedications_07_response_to_medication_index if int(self.nemsis_year) >= 2020 else eMedications_07_response_to_medication_index - 1
    med_dosage_index = eMedications_05_medication_dosage_index if int(self.nemsis_year) >= 2020 else eMedications_05_medication_dosage_index - 1 

    med_set = set()

    write_line_count = 0
    with open(nemsis_med_file_path, "r") as r_f, open(self.cached_si2med_file_path, "w") as w_f:
      for row, line in enumerate(r_f):
        if row == 0:
          continue
        event = line.strip().split('~|~')

        pcr_k = event[pcr_k_index].strip()
        med = event[med_index].strip()
        med_desc = event[med_desc_index].strip()
        response_to_med = event[med_improve_index].strip()
        med_dosage = event[med_dosage_index].strip()

        med_set.add(med)

        if med == '7701001' or med == '7701003' or med == '8801001' or med == '8801003' or med == '8801007' or med == '8801009' or med == '8801019' or med == '8801023':
          continue
        if response_to_med != '9916001':
          continue
        if med_desc == 'Not Applicable':
          continue
        if med_dosage == '7701001' or med_dosage == '7701003':
          continue

        # med_dosage_set.add(med_dosage)

        if pcr_k in nemsis_pcr2si:
          write_line_count += 1
          w_f.write(nemsis_pcr2si[pcr_k] + "~||~" + line)   # temporary split delimiter

    util.writeListFile(self.med_set_file_path, list(med_set))

    print("write %s lines si2med to %s" % (write_line_count, self.cached_si2med_file_path))
    return self.read_cached_si2med(self.cached_si2med_file_path)

if __name__ == "__main__":
    
  time_s = datetime.now()

  parser = argparse.ArgumentParser(description = "control the functions for NEMSIS Medication processing")
  parser.add_argument("--nemsis_dir", action='store', type=str, default = "/slot1/NEMSIS_Databases")
  parser.add_argument("--nemsis_year", action='store', type=str, default = "2020")
  parser.add_argument("--cache_dir", action='store', type=str, default = "nemsis_cache_files")
  args = parser.parse_args()

  mp = Medication_Processor(args.nemsis_dir, args.nemsis_year, args.cache_dir)

  si2med = mp.get_si2med()
  mp.probe_si2med_stat()

  time_t = datetime.now() - time_s
  print("This run takes %s" % time_t)
