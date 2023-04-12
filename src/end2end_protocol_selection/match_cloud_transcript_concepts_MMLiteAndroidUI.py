import os
import pandas as pd
import numpy as np
import math
from datetime import datetime
import Utils as util
from ranking_func import rank
from scipy import spatial
import natsort
import argparse
from threading import Thread
#from vectorize_tamu_protocols import get_dict
#from vectorize_tamu_protocols import vectorizeProtocols


home_dir = "/home/liuyi/Documents/tamu2020fall/mobisys20/EMS-Pipeline/Demo"

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
    res = []
    for line in lines:
        line = line.strip().lower()
        res.append(line)
    f.close()
    return res

def writeListFile(file_path, output_list):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()


def codes_to_texts(codes_lines, refinfo):

  d = refinfo.global_d
  text_lines = []
  word_lines = []
  code_lines = []

  for idx, codes_line in enumerate(codes_lines):
    text_line = []
    word_line = []
    code_line = []

    event = codes_line.split("~|~")
    assert len(event) == 5

    text_line.append(d[event[0]])
    word_line.extend(d[event[0]].split())
    code_line.append(event[0])

    text_line.append(d[event[1]])
    word_line.extend(d[event[1]].split())
    code_line.append(event[1])

    for code in event[2].split(' '):
      text_line.append(d[code])
      word_line.extend(d[code].split())
    code_line.extend(event[2].split(' '))
      
    for code in event[3].split(' '):
      text_line.append(d[code])
      word_line.extend(d[code].split())
    code_line.extend(event[3].split(' '))

    text_lines.append([" ".join(text_line), event[4]])
    word_lines.append([word_line, event[4]])
    code_lines.append([code_line, event[4]])

  return text_lines, word_lines, code_lines

class RefInfo(object):

  def __init__(self):

    nemsis_dir = "/slot1/NEMSIS-files/"
    pri_sym_ref = os.path.join(nemsis_dir, "ESITUATION_09REF.txt")
    pri_imp_ref = os.path.join(nemsis_dir, "ESITUATION_11REF.txt")
    add_sym_ref = os.path.join(nemsis_dir, "ESITUATION_10REF.txt")
    sec_imp_ref = os.path.join(nemsis_dir, "ESITUATION_12REF.txt")

    self.ref_files = [pri_sym_ref, pri_imp_ref, add_sym_ref, sec_imp_ref]
    self.d_list, self.global_d, self.code_map, self.word_map = self.get_dict() 

  def get_dict(self):

    d_list = []
    global_d = dict()

    for ref_idx, ref_file_path in enumerate(self.ref_files):
      ref_f_lines = readFile(ref_file_path)
      ref_f_lines = [s.split("~|~") for s in ref_f_lines]
      d = dict()

      for i, line in enumerate(ref_f_lines):
        if i == 0:
          continue                                                                                                                                                                                      
        
        k = line[0].strip()
        v = line[1].strip()
        v = v.lower()

        if k in d:
          assert d[k] == v
        d[k] = v

        if k in global_d:
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
 

#UVa_concept_set_f = "UVa_concept_set.txt"
def get_dict(word_list):

    word_dict = dict()   # concept - idx dict
    for idx, word in enumerate(word_list):
        word_dict[word.lower()] = idx
    return word_dict

def get_set(f):

    lines = util.readFile(f)
    concept_set = set()
    for idx, line in enumerate(lines):
        concept_set.update(line.split('~|~'))
    concept_list = list(concept_set)
    c_st = natsort.natsorted(concept_list)
    print("obtain %s concepts from %s" % (len(c_st), f))
    return c_st

def getFittedInfo():

    #protocol_csv_file = os.path.join(home_dir, 'Fit_NEMSIS_To_TAMU_Revision1.tsv')
    protocol_csv_file = 'Fit_NEMSIS_To_TAMU_Revision1.tsv'

    # original tamu protocol set size if 108
    tamu_text_f = 'revision1_text.txt'
    tamu_texts = util.readFile(tamu_text_f)

    tamu_text_c_f = 'revision1_text_metamap_concepts.txt'
    tamu_text_c = util.readNEMSISFile(tamu_text_c_f, discard_header = False)

    tamu_text_mmlc_f = 'revision1_text_metamaplite_concepts.txt'
    tamu_text_mmlc = util.readNEMSISFile(tamu_text_mmlc_f, discard_header = False)

    nemsis_id2text = dict()
    nemsis_id2c = dict()
    nemsis_id2mmlc = dict()
    
    protocol_list = pd.read_csv(protocol_csv_file, sep = '\t', dtype = str)
    protocol_list = protocol_list[['TAMU Protocol ID', 'TAMU Protocol', 'NEMSIS Protocol ID', 'NEMSIS Protocol', 'Signs&Symptoms', 'History']]
    protocol_list = protocol_list.dropna()

    idx = 0
    nemsis_id_set = set()
    for _, row in protocol_list.iterrows():

        cur_tamu_id = row['TAMU Protocol ID']
        cur_tamu_name = row['TAMU Protocol']
        cur_nemsis_id = row['NEMSIS Protocol ID']
        cur_nemsis_name = row['NEMSIS Protocol']
        cur_sign = row['Signs&Symptoms']
#        cur_hist = row['History']

#        if cur_tamu_id == "" or cur_tamu_name == "" or cur_nemsis_id == "" or cur_nemsis_name == "" or cur_sign == "" or cur_hist == "":
        if cur_tamu_id == "" or cur_tamu_name == "" or cur_nemsis_id == "" or cur_nemsis_name == "" or cur_sign == "":
            continue

        nemsis_id_set.add(cur_nemsis_id)

        cur_text = tamu_texts[idx]          # str
        cur_c = tamu_text_c[idx]            # list
        cur_mmlc = tamu_text_mmlc[idx]      # list

        if cur_nemsis_id in nemsis_id2c:     # id - concept_list
            assert(isinstance(nemsis_id2c[cur_nemsis_id], list))
            nemsis_id2c[cur_nemsis_id].extend(cur_c)
        else:
            nemsis_id2c[cur_nemsis_id] = cur_c

        if cur_nemsis_id in nemsis_id2mmlc:     # id - concept_list
            assert(isinstance(nemsis_id2mmlc[cur_nemsis_id], list))
            nemsis_id2mmlc[cur_nemsis_id].extend(cur_mmlc)
        else:
            nemsis_id2mmlc[cur_nemsis_id] = cur_mmlc

        if cur_nemsis_id in nemsis_id2text:
            assert(isinstance(nemsis_id2text[cur_nemsis_id], str))
            v = nemsis_id2text[cur_nemsis_id]
            nemsis_id2text[cur_nemsis_id] = v + ' ' + cur_text
        else:
            nemsis_id2text[cur_nemsis_id] = cur_text

        idx += 1

#    print('\nThis curation takes time: %s' % curate_t)

    nemsis_id_list = list(nemsis_id_set)
    nemsis_id_list.sort()
    print("nemsis_id_list size: %s" % len(nemsis_id_list))
    util.writeListFile("fitted_label_names.txt", nemsis_id_list)

    fitted_c_list = []
    fitted_mmlc_list = []
    for nemsis_id in nemsis_id_list:
        c = nemsis_id2c[nemsis_id]          # c is a list
        fitted_c_list.append(c)
        mmlc = nemsis_id2mmlc[nemsis_id]    # mmlc is a list
        fitted_mmlc_list.append(mmlc)

    return fitted_c_list, fitted_mmlc_list, nemsis_id_list

def checkRow(f, vs, row = 0):

    check_row = row
    util.writeListFile(f, map(str, vs[check_row].tolist()))


#
def vectorizeList(concept_dict, concept_list):

#    label_name = concept_list[-1]
#    label_idx = pid_dict[label_name]
#    concept_list = concept_list[:-1]
    encode_count = 0
    # |concepts|
    vector = np.zeros(len(concept_dict), dtype = np.int8)
    for concept in concept_list:           # each concept for one protocol
        if concept in concept_dict:
            encode_count += 1
            c_idx = concept_dict[concept]
            vector[c_idx] += 1
    er = 1.0 * encode_count / len(concept_list)
    return vector, er 

def vectorizeProtocol(concept_dict, concepts_list):
    #  concept_list_f -- the file contains the ~|~ separated concepts for each example
#    concepts_list = util.readNEMSISFile(concept_list_f, discard_header = False)
#    label_ids = []
    # |protocols| x |concepts|
    encode_rates = np.zeros((len(concepts_list)), dtype = np.float16)
    vectors = np.zeros((len(concepts_list), len(concept_dict)), dtype = np.int8)
    for idx, per_id_concept_strs in enumerate(concepts_list):
        per_id_concept_list = []
        for per_id_concept_str in per_id_concept_strs:
            per_id_concept_list.extend(per_id_concept_str.split('~|~'))

#        print(type(concepts))
        vectors[idx], encode_rates[idx] = vectorizeList(concept_dict, per_id_concept_list)
#        label_ids.append(label_id)

    return vectors, encode_rates


def vectorizeNEMSISInput(concept_dict, concept_list_f, label_file, pid_dict):
    #  concept_list_f -- the file contains the ~|~ separated concepts for each example
    concepts_list = util.readNEMSISFile(concept_list_f, discard_header = False)
    label_ids = []
    label_list = util.readFile(label_file)
#    print("concept_list_file: %s, label_file: %s" % (concept_list_f, label_file))
    assert len(label_list) == len(concepts_list)
    # |protocols| x |concepts|
    encode_rates = np.zeros((len(concepts_list)), dtype = np.float16)
    vectors = np.zeros((len(concepts_list), len(concept_dict)), dtype = np.int8)
    for idx, concepts in enumerate(concepts_list):
        label_name = label_list[idx]
        label_id = pid_dict[label_name]
        label_ids.append(label_id)

#        concepts = concepts[:-1]
        vectors[idx], encode_rates[idx] = vectorizeList(concept_dict, concepts)

    return vectors, encode_rates, label_ids

def get_signsymptom2concept_dict(signsymptom_file, concept_file):

    signsymptoms2concepts_dict = dict()
    with open(signsymptom_file, "r") as word_f, open(concept_file, "r") as concept_f:
        for word_line, concept_line in zip(list(word_f), list(concept_f)):
            words = word_line.strip()
            concepts = concept_line.strip()
            signsymptoms2concepts_dict[words] = concepts
    return signsymptoms2concepts_dict

def get_codes2concepts_dict(code2signsymptom_d, signsymptoms2concepts_dict):
    
    codes2concepts_dict = dict()
    for k, v in code2signsymptom_d.items():
        if v not in signsymptoms2concepts_dict:
            print(v)
        assert v in signsymptoms2concepts_dict
        concepts = signsymptoms2concepts_dict[v]
        codes2concepts_dict[k] = concepts
    return codes2concepts_dict

def get_nemsis_input_concept(codes2concepts_dict, nemsis_input_concept_file, refinfo):

    fitted_code_train_file = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_train.txt"
    fitted_code_eval_file = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_eval.txt"
    fitted_code_test_file = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_test.txt"
    _, _, train_code_lines = codes_to_texts(readFile(fitted_code_train_file), refinfo)
    _, _, eval_code_lines = codes_to_texts(readFile(fitted_code_eval_file), refinfo)
    _, _, test_code_lines = codes_to_texts(readFile(fitted_code_test_file), refinfo)
    code_lines = []
    code_lines.extend(train_code_lines)
    code_lines.extend(eval_code_lines)
    code_lines.extend(test_code_lines)
    print("code_lines: %s" % len(code_lines))
    concepts_lines = []
    for code_line in code_lines:
        concepts_line = []
        for code in code_line[0]:
            concept = codes2concepts_dict[code]
            concepts_line.append(concept)
        concepts_line.append(code_line[1])
        line_str = "~|~".join(concepts_line)
        concepts_lines.append(line_str)
#    nemsis_input_mm_concept_file = "nemsis_input_mm_concept.txt"
    writeListFile(nemsis_input_concept_file, concepts_lines)
    print("write %s lines to %s" % (len(concepts_lines), nemsis_input_concept_file))

def get_mixed_concept_set_file(protocol_concept_set_file, nemsis_concept_set_file, protocolandnemsis_concept_set_file):

    f1 = protocol_concept_set_file
    f2 = nemsis_concept_set_file
    f3 = protocolandnemsis_concept_set_file
    lines1 = readFile(protocol_concept_set_file)
    lines2 = readFile(nemsis_concept_set_file)
    writeListFile(f3, lines1 + lines2)
    print("writing %s lines to %s" % (len(lines1 + lines2), f3))

def convert_to_lower_case(f):

    print("converting %s into lowercase" % f)
    lines = readFile(f)
    lower_case_lines = [line.lower() for line in lines]
    writeListFile(f, lower_case_lines)

def generate_input_concept_files(main_args):

    refinfo = RefInfo()
    d_list, global_d, _, _ = refinfo.get_dict()
    sorted_unique_sign_symptom_list = natsort.natsorted(list(set(global_d.values())))
    writeListFile("sorted_unique_sign_symptom_text.txt", sorted_unique_sign_symptom_list)
    
    signsymptoms2mmconcepts_dict = get_signsymptom2concept_dict("sorted_unique_sign_symptom_text.txt", "nemsis_metamap_concepts.txt")
    signsymptoms2mmlconcepts_dict = get_signsymptom2concept_dict("sorted_unique_sign_symptom_text.txt", "nemsis_metamaplite_concepts.txt")
    print("signsymptoms2mmconcepts_dict len: %s, signsymptoms2mmlconcepts_dict len %s" % (len(signsymptoms2mmconcepts_dict), len(signsymptoms2mmlconcepts_dict)))
    
    codes2mmconcepts_dict = get_codes2concepts_dict(global_d, signsymptoms2mmconcepts_dict)
    codes2mmlconcepts_dict = get_codes2concepts_dict(global_d, signsymptoms2mmlconcepts_dict)
    print("codes2mmconcepts_dict len: %s, codes2mmlconcepts_dict len %s" % (len(codes2mmconcepts_dict), len(codes2mmlconcepts_dict)))

#    nemsis_input_mm_concept_file = "nemsis_input_mm_concept.txt"
#    nemsis_input_mml_concept_file = "nemsis_input_mml_concept.txt"

    get_nemsis_input_concept(codes2mmconcepts_dict, main_args.input_mm_file, refinfo)
    get_nemsis_input_concept(codes2mmlconcepts_dict, main_args.input_mml_file, refinfo)

    get_mixed_concept_set_file("revision1_text_metamap_concepts.txt", "nemsis_metamap_concepts.txt", "protocolandnemsis_metamap_concepts.txt")
    get_mixed_concept_set_file("revision1_text_metamaplite_concepts.txt", "nemsis_metamaplite_concepts.txt", "protocolandnemsis_metamaplite_concepts.txt")

    convert_to_lower_case("revision1_text_metamaplite_concepts.txt")

def vectorizeInputConcepts(main_args, mm_concept_f_path, mml_concept_f_path, label_f_path):
#  main_args, mm_concept_f_path, mml_concept_f_path, label_f_path
    fitted_c_list, fitted_mmlc_list, nemsis_id_list = getFittedInfo()
#    print("fitted protocols mm concept list: %s" % (len(fitted_c_list)))
#    print("fitted protocols mmlite concept list: %s" % (len(fitted_mmlc_list)))
#    print("fitted protocols nemsis id list: %s" % (len(nemsis_id_list)))

    metamap_concepts_set_file = None
    metamaplite_concepts_set_file = None
    if main_args.concept_set_source == "protocol":
        metamap_concepts_set_file = "revision1_text_metamap_concepts.txt"
        metamaplite_concepts_set_file = "revision1_text_metamaplite_concepts.txt"
    elif main_args.concept_set_source == "nemsis":
        metamap_concepts_set_file = "nemsis_metamap_concepts.txt"
        metamaplite_concepts_set_file = "nemsis_metamaplite_concepts.txt"
    elif main_args.concept_set_source == "both":
        metamap_concepts_set_file = "protocolandnemsis_metamap_concepts.txt"
        metamaplite_concepts_set_file = "protocolandnemsis_metamaplite_concepts.txt"
    else:
        print("wrong concept_set_source")

    cset_list = get_set(metamap_concepts_set_file)
    mmlcset_list = get_set(metamaplite_concepts_set_file)

    c_dict = get_dict(cset_list)
    mmlc_dict = get_dict(mmlcset_list)

    pid_dict = get_dict(nemsis_id_list)

    c_pvs, c_pers = vectorizeProtocol(c_dict, fitted_c_list)
    mmlc_pvs, mmlc_pers = vectorizeProtocol(mmlc_dict, fitted_mmlc_list)

    print("\n=============== vectorizing ====================")
    print("concept set files:", metamap_concepts_set_file, metamaplite_concepts_set_file)

    print("(metamap) tamu protocol array shape: %s, encode rate: %s" % (c_pvs.shape, np.average(c_pers)))
    print("(metamaplite) tamu protocol array shape: %s, encode rate: %s" % (mmlc_pvs.shape, np.average(mmlc_pers)))
#    print("nemsis metamap set, tamu protocol, encode rate: ", np.average(c_pers))
#    print("protocol: average tamu metamaplite set vectorize tamu protocol encode rate: ", np.average(mmlc_pers))

    c_ivs, c_iers, c_ilabel_ids = vectorizeNEMSISInput(c_dict, mm_concept_f_path, label_f_path, pid_dict)
    mmlc_ivs, mmlc_iers, mmlc_ilabel_ids = vectorizeNEMSISInput(mmlc_dict, mml_concept_f_path, label_f_path, pid_dict)

    print("(metamap) given input array shape: %s, encode rate: %s" % (c_ivs.shape, np.average(c_iers)))
    print("(metamaplite) given input array shape: %s, encode rate: %s" % (mmlc_ivs.shape, np.average(mmlc_iers)))
#    print("nemsis metamap set, filtered nemsis, encode rate: ", np.average(c_iers))
#    print("input: average filtered nemsis metamaplite encode rate: ", np.average(mmlc_iers))

#    return c_pvs, c_ivs, c_ilabel_ids
    return c_pvs, mmlc_pvs, c_ivs, mmlc_ivs, c_ilabel_ids, mmlc_ilabel_ids
   
def computeSimilarity(tv, pvs, pids, metric = "cosine"):

    ranking_list = []
    for i in range(pvs.shape[0]):
        pv = pvs[i]
        cur_id = pids[i]
        sim = 0.0
        if metric == "cosine":
            sim = 1 - spatial.distance.cosine(tv, pv)
            if math.isnan(sim):
                sim = 0.0
        elif metric == "dot":
            sim = np.dot(tv, pv)
        else:
            print("we require a metric: cosine, dot_product")
            exit(1)
            
        ranking_list.append((cur_id, sim))
        
    return ranking_list

def get_top_score(pvs, tvs, pids, tids, main_args):
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0   
    total_count = tvs.shape[0]
    for idx in range(tvs.shape[0]):

        if idx != 94:
            continue

        tv = tvs[idx]
        tp = tids[idx]
        ranking = computeSimilarity(tv, pvs, pids, metric = main_args.metric)
        # rank
        candi_list, score_list = rank(ranking)[0], rank(ranking)[1]
        top1set = set()
        top3set = set()
        top5set = set()
        # for each input, we get an top-1, top-3, top-5 score, respectively
        num = sum(score_list[:5])
        for candi_idx, candi_id in enumerate(candi_list):
#            candi_id_st = set(candi_id.split(';'))
            if candi_idx >= 5:
                break
            candi_score = (0.0 if num == 0.0 else score_list[candi_idx] / num)
            if candi_idx == 0:
                top1set.add(candi_id)
            if candi_idx <= 2:
                top3set.add(candi_id)
            top5set.add(candi_id)
            print("index: %s, top-%s, candidate_id: %s, candidate score: %s" % (idx, candi_idx, candi_id, candi_score))
#            out_count += 1
        exit(1)
        if tp in top1set != 0:
            top1 += 1.0

        if tp in top3set != 0:
            top3 += 1.0

        if tp in top5set != 0:
            top5 += 1.0

    return top1/total_count, top3/total_count, top5/total_count
        
def protocol_selection(main_args, mm_concept_f_path, mml_concept_f_path, label_f_path):
   

    c_pvs, mmlc_pvs, c_ivs, mmlc_ivs, c_ilabel_ids, mmlc_ilabel_ids = vectorizeInputConcepts(main_args, mm_concept_f_path, mml_concept_f_path, label_f_path)

    print("\n=============== protocol_selection ====================")
    pids = range(c_pvs.shape[0])
    c_top1, c_top3, c_top5 = get_top_score(c_pvs, c_ivs, pids, c_ilabel_ids, main_args)
    print("(cosine) input metamap topk: ", c_top1, c_top3, c_top5)
    mmlc_top1, mmlc_top3, mmlc_top5 = get_top_score(mmlc_pvs, mmlc_ivs, pids, mmlc_ilabel_ids, main_args)
    print("(cosine) input metamaplite topk: ", mmlc_top1, mmlc_top3, mmlc_top5)

    return c_top1, c_top3, c_top5, mmlc_top1, mmlc_top3, mmlc_top5

def match_for_spk_dir(main_args, spk_path, spk_dir):
    
    # different from concepts generation, the code for all concepts are the same: cloud_transcription/true_text
    files = [
        "eval_GC_transcript_command_and_search.txt",
        "eval_GC_transcript_default.txt",
        "eval_GC_transcript_latest_long.txt",
        "eval_GC_transcript_latest_short.txt",
        "eval_GC_transcript_medical_conversation.txt",
        "eval_GC_transcript_medical_dictation.txt",
        "eval_GC_transcript_phone_call.txt",
        "eval_GC_transcript_video.txt",
        "sampled_signs_symptoms_100.txt"
    ]
    
    acc_lines = []
    for idx, f in enumerate(files):
        if idx != 8:
            continue
        f_base_name = f.split(".")[0]
        mm_concept_f_path = os.path.join(spk_path, f_base_name + "_concepts.txt")
        mml_concept_f_path = os.path.join(spk_path, f_base_name + "_mml_concepts.txt")
        label_f_path = os.path.join(spk_path, "labels.txt")
        assert os.path.isfile(mm_concept_f_path)
        assert os.path.isfile(mml_concept_f_path)
        assert os.path.isfile(label_f_path)
        
#        c_top1, c_top3, c_top5, mmlc_top1, mmlc_top3, mmlc_top5 = protocol_selection(main_args, mm_concept_f_path, mml_concept_f_path, label_f_path)
        acc_line = []
        top_k_acc = protocol_selection(main_args, mm_concept_f_path, mml_concept_f_path, label_f_path)
        for acc in top_k_acc:
            acc_line.append(str(acc))
        assert len(acc_line) == 6
        acc_lines.append(",".join(acc_line))

    assert len(acc_lines) == 9
    acc_csv_name = os.path.join(spk_path, "concept_match_" + main_args.metric + ".csv")
    util.writeListFile(acc_csv_name, acc_lines)
    print("write %s lines to %s" % (len(acc_lines), acc_csv_name))

def match_for_dir(main_args):

    main_dir = main_args.main_dir
    spk_dirs = os.listdir(main_dir)
    assert len(spk_dirs) == 7
    spk_paths = []
    for spk_dir in spk_dirs:
        if spk_dir != "sample_liuyi":
            continue
        spk_path = os.path.join(main_dir, spk_dir)
        spk_paths.append(spk_path)

    threads_list = []
    for thread_num in range(len(spk_paths)):
        print("match for speaker %s" % spk_paths[thread_num])
        tmp_t = Thread(target = match_for_spk_dir, args=(main_args, spk_paths[thread_num], spk_dirs[thread_num]))
        threads_list.append(tmp_t)
        tmp_t.start()
        print("thread %s start!" % thread_num)
    for t in threads_list:
        t.join()

if __name__ == "__main__":

    t_start = datetime.now()  

    parser = argparse.ArgumentParser(description = "control the functions for EMSBert")
    parser.add_argument("--regen", action='store_true', default=False, help="indicate whether to regenerate files for concept matching")
    parser.add_argument("--concept_set_source", action='store', type=str, help="protocol, nemsis, both", required=True)
#    parser.add_argument("--input_mm_file", action='store', type=str, default="nemsis_input_mm_concept.txt", help="protocol, nemsis, both", required=True)
#    parser.add_argument("--input_mml_file", action='store', type=str, default="nemsis_input_mml_concept.txt", help="protocol, nemsis, both", required=True)
    parser.add_argument("--metric", action='store', type=str, help="cosine, dot", required=True)

    parser.add_argument("--main_dir", action='store', type=str, default="/home/liuyi/MetamapMatching/google_cloud")


    main_args = parser.parse_args()
    
    if main_args.regen:
        generate_input_concept_files()

    match_for_dir(main_args)
    t_total = datetime.now() - t_start
    print("\nThis run takes time: %s" % t_total) 
