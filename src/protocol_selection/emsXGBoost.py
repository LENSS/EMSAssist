import collections
import argparse
import tensorflow as tf
import numpy as np
from tensorflow import keras
from datetime import datetime
import tokenization
import optimization
import os
import csv
import random
from absl import logging
import natsort
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.metrics import top_k_accuracy_score
import file_util


logging.set_verbosity(logging.INFO)

global_seed = 1993

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

def writeListFile(file_path, output_list):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               guid,
               text:str,
               words:list,
               codes:list,
               label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
      weight: (Optional) float. The weight of the example to be used during
        training.
      int_iden: (Optional) int. The int identification number of example in the
        corpus.
    """
    self.guid = guid
    self.text = text
    self.words = words
    self.codes = codes
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               label_id):
    self.input_ids = input_ids
    self.label_id = label_id

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
#    print("total codes %s" % len(global_d))
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
 
def counting_dict(list_to_count: list):

  count_d = collections.OrderedDict()
#  print(type(count_d))
  for e in list_to_count:
    if e in count_d:
      count_d[e] += 1
    else:
      count_d[e] = 1
  return count_d

def to_scipy_sparse(sparse_indices, data, dense_shape):

  row = []
  col = []
  for indices in sparse_indices:
    assert len(indices) == 2
    row.append(indices[0])
    col.append(indices[1])
  return csr_matrix((data, (row, col)), dense_shape)

def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 feature_type,
                                 refinfo):
  """Convert a set of `InputExample`s to a TFRecord file."""

  assert feature_type == "token_counts" or feature_type == "tokens" or feature_type == "word_counts" or feature_type == "words" or feature_type == "code_counts" or feature_type == "codes"
#  features = []
  sparse_indices = []
  sparse_values = []
  label_ids = []

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  if feature_type == "token_counts" or feature_type == "tokens":
    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        logging.info("converting example %d of %d", ex_index, len(examples))

      tokens = tokenizer.tokenize(example.text)
      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("tokens: %s",
                     " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])
#      
      col_index_cnt = counting_dict(input_ids)
      for col_index, cnt in col_index_cnt.items():
        sparse_indices.append([ex_index, col_index])
        if feature_type == "token_counts":
#          print("You chose token_counts as your feature")
          sparse_values.append(cnt)
        elif feature_type == "tokens":
#          print("You chose tokens as your feature")
          sparse_values.append(1)
        else:
          print("Please choose right feature_type")
      label_ids.append(label_map[example.label])
      
#      if ex_index == 0:
#
#        print("sparse_indices:")
#        print(sparse_indices)
#        print("sparse_values:")
#        print(sparse_values)
#        print("scipy_presentation:")
#        print(to_scipy_sparse())
#
#        csr_matrix((data, (row, col)), dense_shape)

#    return tf.sparse.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape = [len(features), len(tokenizer.vocab)]), label_ids 

    return to_scipy_sparse(sparse_indices, sparse_values, (len(examples), len(tokenizer.vocab))), label_ids

  elif feature_type == "word_counts" or feature_type == "words":
    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        logging.info("converting example %d of %d", ex_index, len(examples))

      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("words: %s", ", ".join([str(x) for x in example.words]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])
        #logging.info("label: %s (id = %s)", example.label, label_map(example.label))

      col_index_cnt = counting_dict(example.words)
      for word, cnt in col_index_cnt.items():
        sparse_indices.append([ex_index, refinfo.word_map[word]])
        if feature_type == "word_counts":
#          print("You chose word_counts as your feature")
          sparse_values.append(cnt)
        elif feature_type == "words":
#          print("You chose words as your feature")
          sparse_values.append(1)
        else:
          print("Please choose right feature_type")
      label_ids.append(label_map[example.label])

#    return tf.sparse.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape = [len(examples), len(refinfo.word_map)]), label_ids 
    return to_scipy_sparse(sparse_indices, sparse_values, (len(examples), len(refinfo.word_map))), label_ids
      
  elif feature_type == "code_counts" or feature_type == "codes":
    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        logging.info("converting example %d of %d", ex_index, len(examples))

      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("codes: %s", ", ".join([str(x) for x in example.codes]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])
        #logging.info("label: %s (id = %s)", example.label, label_map(example.label))

      col_index_cnt = counting_dict(example.codes)
      for code, cnt in col_index_cnt.items():
        sparse_indices.append([ex_index, refinfo.code_map[code]])
        if feature_type == "code_counts":
#          print("You chose code_counts as your feature")
          sparse_values.append(cnt)
        elif feature_type == "codes":
#          print("You chose codes as your feature")
          sparse_values.append(1)
        else:
          print("Please choose right feature_type")
      label_ids.append(label_map[example.label])

    #return tf.sparse.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape = [len(examples), len(refinfo.code_map)]), label_ids 
    #sparse_features = to_scipy_sparse(sparse_indices, sparse_values, (len(examples), len(refinfo.code_map)))
    #return sparse_features, label_ids
    #print("total sparse values: %s" % len(sparse_values))
    return to_scipy_sparse(sparse_indices, sparse_values, (len(examples), len(refinfo.code_map))), label_ids

def build_vocab_tokenizer(vocab_file, do_lower_case):

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
  return tokenizer

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


def from_codes(filename,
             text_column,
             label_column,
             do_lower_case,
             shuffle,
             vocab_file,
             feature_type,
             is_training,
             refinfo,
             fieldnames=None,
             delimiter=',',
             quotechar='"'):
  """Loads text with labels from the csv file and preproecess text according to `model_spec`.
  Args:
    filename: Name of the file.
    text_column: String, Column name for input text.
    label_column: String, Column name for labels.
    fieldnames: A sequence, used in csv.DictReader. If fieldnames is omitted,
      the values in the first row of file f will be used as the fieldnames.
    model_spec: Specification for the model.
    is_training: Whether the loaded data is for training or not.
    delimiter: Character used to separate fields.
    quotechar: Character used to quote fields containing special characters.
    shuffle: boolean, if shuffle, random shuffle data.
    cache_dir: The cache directory to save preprocessed data. If None,
      generates a temporary directory to cache preprocessed data.
  Returns:
    TextDataset containing text, labels and other related info.
  """
  csv_name = os.path.basename(filename)

  tokenizer = build_vocab_tokenizer(vocab_file, do_lower_case)

  assert len(tokenizer.vocab) == len(tokenizer.inv_vocab)

  lines = readFile(filename)
  text_lines, word_lines, code_lines = codes_to_texts(lines, refinfo)

  # Gets labels.
  label_set = set()
  for line in text_lines:
    label_set.add(line[1])
  label_names = sorted(label_set)

  # Generates text examples from csv file.
  examples = []
  for i, (text_line, word_line, code_line) in enumerate(zip(text_lines, word_lines, code_lines)):
#    if i == 0:
#        print(line[0], line[1])
    text, label = text_line[0], text_line[1]
    words, label = word_line[0], word_line[1]
    codes, label = code_line[0], code_line[1]
    guid = '%s-%d' % (csv_name, i)
    examples.append(InputExample(guid, text, words, codes, label))

  meta_data = {
      'size': len(examples),
      'num_classes': len(label_names),
      'index_to_label': label_names,
#      'input_feature_dim': len(tokenizer.vocab)
  }

  if feature_type == "tokens" or feature_type == "token_counts":
    meta_data['input_faeture_dim'] = len(tokenizer.vocab)
  elif feature_type == "words" or feature_type == "word_counts":
    meta_data['input_feature_dim'] = len(refinfo.word_map)
  elif feature_type == "codes" or feature_type == "code_counts":
    meta_data['input_feature_dim'] = len(refinfo.code_map)

  random.seed(global_seed)
  if shuffle:
    random.shuffle(examples)

  sparse_features, label_ids = convert_examples_to_features(examples, label_names, tokenizer, feature_type, refinfo)
  return sparse_features, np.array(label_ids), meta_data

def prepare_dataset(args, refinfo):


    test_file_name = os.path.join(args.eval_dir, args.test_file)
    test_x, test_y, test_meta_data = from_codes(
          filename=test_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=False,
          shuffle=False,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo)
    print(test_meta_data)

    if args.test_only:
        return None, None, None, None, None, None, test_x, test_y, test_meta_data

    train_file_name = os.path.join(args.eval_dir, args.train_file)
    train_x, train_y, train_meta_data = from_codes(
          filename=train_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=True,
          shuffle=True,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo)
    print(train_meta_data)

    eval_file_name = os.path.join(args.eval_dir, args.eval_file)
    eval_x, eval_y, eval_meta_data = from_codes(
          filename=eval_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=False,
          shuffle=False,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo)
    print(eval_meta_data)

    return train_x, train_y, train_meta_data, eval_x, eval_y, eval_meta_data, test_x, test_y, test_meta_data

def top1_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 1, name = 'top1', dtype=tf.float32)

def top3_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 3, name = 'top3', dtype=tf.float32)

def top5_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 5, name = 'top5', dtype=tf.float32)

def calculate_top_k(test_Y_pred, test_Y):

    # calculate top1, top3, tyop5 accuracy
    top1_acc = 0
    top3_acc = 0
    top5_acc = 0
    for i in range(len(test_Y)):
        if test_Y[i] == np.argmax(test_Y_pred[i]):
            top1_acc += 1
        if test_Y[i] in np.argsort(test_Y_pred[i])[-3:]:
            top3_acc += 1
        if test_Y[i] in np.argsort(test_Y_pred[i])[-5:]:
            top5_acc += 1
    print("top1 accuracy: %s" % (top1_acc / len(test_Y)))
    print("top3 accuracy: %s" % (top3_acc / len(test_Y)))
    print("top5 accuracy: %s" % (top5_acc / len(test_Y)))

def test_best_model(clf, test_X, test_Y, params):

#    print("test with the optimal the round(tree)")
    #if hasattr(clf, 'best_iteration'):
    if 'best_iteration' in params:
        #print("out of %s boosting rounds (trees), %s-th is the best round" % (clf.get_num_boosting_rounds(), clf.best_iteration))
        print("best_iteration: %s " % params['best_iteration'])
        test_Y_pred = clf.predict_proba(test_X, ntree_limit= params['best_iteration'])
        calculate_top_k(test_Y_pred, test_Y)

    #if hasattr(clf, 'best_ntree_limit'):
    if 'best_ntree_limit' in params:
        print("best_ntree_limit: %s" % params['best_ntree_limit'])
        test_Y_pred = clf.predict_proba(test_X, ntree_limit= params['best_ntree_limit'])
        calculate_top_k(test_Y_pred, test_Y)
    
    print("evaluate on test set:")

    test_Y_pred = clf.predict_proba(test_X) 
    calculate_top_k(test_Y_pred, test_Y)

def train_xgboost_model(prepared_data, args):

    train_X, train_Y, train_meta_data, val_X, val_Y, eval_meta_data, test_X, test_Y, test_meta_data = prepared_data 
#    train_d = xgb.DMatrix(train_X, label=train_Y)
#    val_d = xgb.DMatrix(val_X, label=val_Y)
#    test_d = xgb.DMatrix(test_X, label=test_Y)
#    num_classes = train_meta_data['num_classes']
#    num_round = 128
#    params = file_util.load_json_file(model_params_path)
#
#    params["num_class"] = train_meta_data['num_classes']
#    params['n_estimators'] = args.num_round
#    params['early_stopping_rounds'] = args.early_stopping_rounds
#    params['random_state'] = global_seed
#
    if not tf.io.gfile.exists(args.model_dir):
      print("we will save xgboost to %s" % args.model_dir)
      tf.io.gfile.makedirs(args.model_dir)

    params = {'max_depth': 20,
            'colsample_bylevel': 0.7, 
            'max_leaves': 0,
            'learning_rate': args.lr,
            'gpu_id': 0,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            'verbosity': 1,
            'random_state': global_seed,
#            'enable_categorical': 'True', 
            #'silent': 1, 
            'objective': 'multi:softprob', 
            #'eval_metric': ['mlogloss','auc'],
            'eval_metric': ['mlogloss'],
            #'eval_metric':[top_k_accuracy_score],
            'num_class': train_meta_data['num_classes'],
#            'verbose_eval': 128,
            'n_estimators': args.num_round,
            'early_stopping_rounds': args.early_stopping_rounds,
#            'verbose_eval': args.verbose_eval
            }


#    bst = xgb.train(params, train_d, num_boost_round = num_round, evals=[(val_d,'eval'),(test_d, 'test')], verbose_eval=128)

    clf = xgb.XGBClassifier(
        **params,
#        # eval_metric="auc",
#        # enable_categorical=True,
#        # max_cat_to_onehot=1,  # We use optimal partitioning exclusively
    )

#    print(clf.get_params())
#    print(clf.get_xgb_params())

#    print("out of %s boosting rounds (trees)" % clf.get_num_boosting_rounds())
    print(params)

    clf.fit(train_X, train_Y, eval_set=[(val_X, val_Y)], early_stopping_rounds= params['early_stopping_rounds'])
    #clf.fit(train_X, train_Y, eval_set=[((train_X, train_Y), "train"), ((val_X, val_Y), "validate")])
    #clf.fit(train_X, train_Y, eval_set=[(train_X, train_Y), (val_X, val_Y)])

    saved_model_path = os.path.join(args.model_dir, "emsXGBoost.json")
    print("saving emsXGBoost to %s" % saved_model_path)
    clf.save_model(saved_model_path)

    if hasattr(clf, 'best_iteration'):
        params['best_iteration'] = clf.best_iteration
    if hasattr(clf, 'best_ntree_limit'):
        params['best_ntree_limit'] = clf.best_ntree_limit

    model_params_path = os.path.join(args.model_dir, "params.json")
    file_util.write_json_file(model_params_path, params)
#    print(clf.get_params())

    test_best_model(clf, test_X, test_Y, params)

def test_xgboost_model(prepared_data, args):

    train_X, train_Y, train_meta_data, val_X, val_Y, eval_meta_data, test_X, test_Y, test_meta_data = prepared_data 
    model_params_path = os.path.join(args.test_model_path, "params.json")
    params = file_util.load_json_file(model_params_path)

    clf = xgb.XGBClassifier(
        **params,
    )
    clf.load_model(os.path.join(args.test_model_path, "emsXGBoost.json"))
    print(params)

    test_best_model(clf, test_X, test_Y, params)

if __name__ == "__main__":
    
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for EMSBert")
    parser.add_argument("--eval_dir", action='store', type=str, default = "data", help="directory containing resources for specific purposes")
    parser.add_argument("--train_file", action='store', type=str, default="train.tsv", help="train file name")
    parser.add_argument("--eval_file", action='store', type=str, default="eval.tsv", help="eval file name")
    parser.add_argument("--test_file", action='store', type=str, default="test.tsv", help="test file name")
    parser.add_argument("--vocab_file", action='store', type=str, default="/home/liuyi/emsNet/vocab.txt", help="test file name")
    parser.add_argument("--cuda_device", action='store', type=str, default = "1", help="indicate the cuda device number")
    parser.add_argument("--feature_type", action='store', type=str, default = "tokens", help="indicate the feature type")
    parser.add_argument("--model_dir", action='store', type=str, default = 'saved_model', help = "indicate where to store the trained models")
    parser.add_argument("--test_only", action='store_true', default=False, help="indicate whether to only do testing")
    parser.add_argument("--num_round", type=int, default=1280, help="total boosting rounds")
    parser.add_argument("--early_stopping_rounds", type=int, default=10, help="total boosting rounds")
#    parser.add_argument("--verbose_eval", type=int, default=128, help="printing frequency in boosting rounds")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    parser.add_argument("--test_model_path", action='store', type=str, help = "indicate where to store the trained models", required = True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    refinfo = RefInfo()
    #train_ds, train_meta_data, validation_ds, eval_meta_data, test_ds, test_meta_data = prepare_dataset(args)
    prepared_data = prepare_dataset(args, refinfo)

    #train_ann_model(prepared_data, args)
    if args.test_only:
        test_xgboost_model(prepared_data, args)
    else:
        train_xgboost_model(prepared_data, args)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)
