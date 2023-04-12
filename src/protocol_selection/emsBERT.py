import numpy as np
import os
import collections
import csv
from absl import logging
logging.set_verbosity(logging.INFO)
import tensorflow as tf
print(tf.__version__)
assert tf.__version__.startswith('2')
import tensorflow_addons as tfa

import file_util
import tokenization
import optimization
import configs as bert_configs
import quantization_configs as quant_configs
import metadata_writer_for_bert_text_classifier as bert_metadata_writer
import tensorflow_hub as hub
from datetime import datetime
import random
import tempfile
import argparse
import natsort
# import time

#import TFUtils as tfutil
# import pandas as pd
#import Utils as util

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

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               guid,
               text_a,
               text_b=None,
               label=None,
               int_iden=None):
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
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.int_iden = int_iden


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True,
               int_iden=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    self.int_iden = int_iden

class RefInfo(object):

  def __init__(self):

    nemsis_dir = "../../data/ae_text_data/NEMSIS_files/"
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
 

def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example

def single_file_dataset(input_file, name_to_features):
  """Creates a single-file dataset to be passed for BERT custom training."""
  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  d = d.map(
      lambda record: decode_record(record, name_to_features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # When `input_file` is a path to a single file or a list
  # containing a single path, disable auto sharding so that
  # same input file is sent to all workers.
  if isinstance(input_file, str) or len(input_file) == 1:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    d = d.with_options(options)
  return d

def get_name_to_features(seq_len):
  """Gets the dictionary describing the features."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_len], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64),
      'is_real_example': tf.io.FixedLenFeature([], tf.int64),
  }
  return name_to_features

def get_name_to_features_tflite(seq_len):

  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([1, seq_len], tf.int64),
      'input_mask': tf.io.FixedLenFeature([1, seq_len], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([1, seq_len], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64),
      'is_real_example': tf.io.FixedLenFeature([], tf.int64),
  }
  return name_to_features

def select_data_from_record(record):
  """Dispatches records to features and labels."""
  x = {
      'input_word_ids': record['input_ids'],
      'input_mask': record['input_mask'],
      'input_type_ids': record['segment_ids']
  }
  y = record['label_ids']
  return (x, y)

def _load(tfrecord_file, meta_data_file, max_seq_len, test_tflite=False):
  """Loads data from tfrecord file and metada file."""

  name_to_features = get_name_to_features(max_seq_len)

  if test_tflite:
    name_to_features = get_name_to_features_tflite(max_seq_len)

  dataset = single_file_dataset(tfrecord_file, name_to_features)
  dataset = dataset.map(select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)

  meta_data = file_util.load_json_file(meta_data_file)

  logging.info(
      'Load preprocessed data and metadata from %s and %s '
      'with size: %d', tfrecord_file, meta_data_file, meta_data['size'])
  return dataset, meta_data

def get_cache_info(cache_dir, data_name):

  assert(cache_dir is not None)
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)

  file_prefix = data_name.split('.')[0]
  cache_prefix = os.path.join(cache_dir, file_prefix)
  tfrecord_file = cache_prefix + '.tfrecord'
  meta_data_file = cache_prefix + '_meta_data'
  is_cached = tf.io.gfile.exists(tfrecord_file) and tf.io.gfile.exists(meta_data_file)

  return is_cached, tfrecord_file, meta_data_file

#def read_csv(input_file, fieldnames=None, delimiter=',', quotechar='"'):
#  """Reads a separated value file."""
#  with tf.io.gfile.GFile(input_file, 'r') as f:
#    reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
#    lines = []
#    for line in reader:
#      lines.append(line)
#    return lines

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  if label_list:
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label] if label_map else example.label
  if ex_index < 5:
    logging.info("*** Example ***")
    logging.info("guid: %s", (example.guid))
    logging.info("tokens: %s",
                 " ".join([tokenization.printable_text(x) for x in tokens]))
    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    logging.info("label: %s (id = %s)", example.label, str(label_id))
    logging.info("int_iden: %s", str(example.int_iden))

  feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True,
        int_iden=example.int_iden)

  return feature


def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_file,
                                            label_type=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if label_type is not None and label_type == float:
      features["label_ids"] = create_float_feature([feature.label_id])
    elif feature.label_id is not None:
      features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    if feature.int_iden is not None:
      features["int_iden"] = create_int_feature([feature.int_iden])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _save_data(examples, label_names, max_seq_len, tokenizer, tfrecord_file, meta_data_file):
  """Saves preprocessed data and other assets into files."""
  # Converts examples into preprocessed features and saves in tfrecord_file.
  file_based_convert_examples_to_features(examples, label_names, max_seq_len, tokenizer, tfrecord_file)

  # Generates and saves meta data in meta_data_file.
  meta_data = {
      'size': len(examples),
      'num_classes': len(label_names),
      'index_to_label': label_names
  }
  file_util.write_json_file(meta_data_file, meta_data)

def build_vocab_tokenizer(model_uri, do_lower_case):
  """Builds the class. Used for lazy initialization."""
  vocab_file = os.path.join(model_uri, 'assets', 'vocab.txt')
  assert(tf.io.gfile.exists(vocab_file))

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
  return vocab_file, tokenizer

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
             model_uri,
             max_seq_len,
             refinfo,
             do_lower_case=True,
             fieldnames=None,
             is_training=True,
             delimiter=',',
             quotechar='"',
             shuffle=False,
             cache_dir=None,
             test_tflite=False):
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

  is_cached, tfrecord_file, meta_data_file = get_cache_info(cache_dir, csv_name)

  # print("tfrecord file: %s" % tfrecord_file)

  # If cached, directly loads data from cache directory.
  # if is_cached:
  #   return _load(tfrecord_file, meta_data_file, max_seq_len, test_tflite=test_tflite)


  vocab_file, tokenizer = build_vocab_tokenizer(model_uri, do_lower_case)

#  lines = read_csv(filename, fieldnames, delimiter, quotechar)
  random.seed(global_seed)
  lines = readFile(filename)
  text_lines, word_lines, code_lines = codes_to_texts(lines, refinfo)
  if shuffle:
    random.shuffle(text_lines)

  # Gets labels.
  label_set = set()
  for line in text_lines:
    label_set.add(line[1])
  label_names = sorted(label_set)

  # Generates text examples from csv file.
  examples = []

  for i, line in enumerate(text_lines):
    text, label = line[0], line[1]
    guid = '%s-%d' % (csv_name, i)
    examples.append(InputExample(guid, text, None, label))

  # Saves preprocessed data and other assets into files.
  _save_data(examples, label_names, max_seq_len, tokenizer, tfrecord_file, meta_data_file)

  # Loads data from cache directory.
  return _load(tfrecord_file, meta_data_file, max_seq_len, test_tflite=test_tflite)

def gen_dataset(dataset,
                batch_size=1,
                is_training=False,
                shuffle=False,
                drop_remainder=False):
  """Generate a shared and batched tf.data.Dataset for training/evaluation.
  Args:
    batch_size: A integer, the returned dataset will be batched by this size.
    is_training: A boolean, when True, the returned dataset will be optionally
      shuffled and repeated as an endless dataset.
    shuffle: A boolean, when True, the returned dataset will be shuffled to
      create randomness during model training.
    input_pipeline_context: A InputContext instance, used to shared dataset
      among multiple workers when distribution strategy is used.
    preprocess: A function taking three arguments in order, feature, label and
      boolean is_training.
    drop_remainder: boolean, whether the finaly batch drops remainder.
  Returns:
    A TF dataset ready to be consumed by Keras model.
  """
  ds = dataset
  if is_training:
    if shuffle:
      buffer_size = 3 * batch_size
      ds = ds.shuffle(buffer_size=buffer_size)

  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

# Step 2. Load the training and test data, then preprocess them according to a specific model_spec.

def prepare_dataset(args, refinfo):

    #if args.test_tflite:
    #    return None, None, None, None, None, None 

    test_file_name = os.path.join(args.eval_dir, args.test_file)
    test_data, test_meta_data = from_codes(
          filename=test_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          delimiter='\t',
          is_training=False,
          cache_dir=args.eval_dir,
          model_uri=args.test_model_path,
          max_seq_len=args.max_seq_len,
          shuffle=False,
          refinfo=refinfo, 
          test_tflite=args.test_tflite)
    test_ds = gen_dataset(test_data, args.test_batch_size, is_training=False)
    print(test_meta_data)

    if args.do_test or args.test_tflite:
      return None, None, None, None, test_ds, test_meta_data    

    train_file_name = os.path.join(args.eval_dir, args.train_file)
    train_data, train_meta_data = from_codes(
          filename=train_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          delimiter='\t',
          is_training=True,
          cache_dir=args.eval_dir,
          model_uri=args.init_model,
          max_seq_len=args.max_seq_len,
          shuffle=True, 
          refinfo=refinfo)
    train_ds = gen_dataset(train_data, args.train_batch_size, is_training=True)
    print(train_meta_data)
    
    eval_file_name = os.path.join(args.eval_dir, args.eval_file)
    eval_data, eval_meta_data = from_codes(
          filename=eval_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          delimiter='\t',
          is_training=False,
          cache_dir=args.eval_dir,
          model_uri=args.init_model,
          max_seq_len=args.max_seq_len,
          shuffle=False,
          refinfo=refinfo)
    validation_ds = gen_dataset(eval_data, args.eval_batch_size, is_training=False)
    print(eval_meta_data)
   
    return train_ds, train_meta_data, validation_ds, eval_meta_data, test_ds, test_meta_data

def create_classifier_model(bert_config,
                            num_labels,
                            max_seq_length,
                            initializer=None,
                            hub_module_url=None,
                            hub_module_trainable=True,
                            is_tf2=True):
  """BERT classifier model in functional API style.
  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.
  Args:
    bert_config: BertConfig, the config defines the core Bert model.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    initializer: Initializer for the final dense layer in the span labeler.
      Defaulted to TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.
    is_tf2: boolean, whether the hub module is in TensorFlow 2.x format.
  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  if initializer is None:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range, seed=global_seed)

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

  print("bert_trainable: %s" % hub_module_trainable)

  bert_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
  model_outputs = bert_model({
      'input_word_ids': input_word_ids,
      'input_mask': input_mask,
      'input_type_ids': input_type_ids
  })
  pooled_output = model_outputs['pooled_output']
  
#  pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
  output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)
  print("output matrix dimension of mobilebert:")
  print(output.shape)

  output = tf.keras.layers.Dense(
      num_labels,
      kernel_initializer=initializer,
      name='output',
      activation='softmax',
      dtype=tf.float32)(
          output)


  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=output), bert_model

def create_model(num_classes,
                 optimizer='adam',
                 with_loss_and_metrics=True,
                 dropout_rate=0.1,
                 args=None):
  """Creates the keras model."""
  bert_config = bert_configs.BertConfig(
        0,
        initializer_range = 0.02,
        hidden_dropout_prob = dropout_rate)
  bert_model, _ = create_classifier_model(
      bert_config,
      num_classes,
      max_seq_length = args.max_seq_len,
      hub_module_url=args.init_model,
      hub_module_trainable=args.bert_trainable,
#      hub_module_trainable=True,
      is_tf2=True)

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  def top1_metric_fn():
    return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 1, name = 'top1_accuracy', dtype=tf.float32)

  def top3_metric_fn():
    return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 3, name = 'top3_accuracy', dtype=tf.float32)

  def top5_metric_fn():
    return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 5, name = 'top5_accuracy', dtype=tf.float32)


  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
#  def metric_fn():
#    return tf.keras.metrics.SparseCategoricalAccuracy(
#        'test_accuracy', dtype=tf.float32)

  if with_loss_and_metrics:
    # Add loss and metrics in the keras model.
    bert_model.compile(
        optimizer=optimizer,
#        loss=tfa.losses.SigmoidFocalCrossEntropy(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[top1_metric_fn(), top3_metric_fn(), top5_metric_fn()])

  return bert_model

def prepare_model(train_ds, train_meta_data, validation_ds, args):
    
    steps_per_epoch = train_meta_data['size'] // args.train_batch_size
    train_ds = train_ds.take(steps_per_epoch)
    total_steps = steps_per_epoch * args.train_epoch
    logging.info("total training steps: %s" % total_steps)
    warmup_steps = int(total_steps * 0.01)
    initial_lr = 3e-5
    dropout_rate=0.1
    seq_len=args.max_seq_len
    num_classes = train_meta_data['num_classes']
    model_dir = args.model_dir
    optimizer = optimization.create_optimizer(initial_lr, total_steps, warmup_steps)
    bert_model = create_model(num_classes, optimizer, dropout_rate=dropout_rate, args=args)

    print("%s saved models will be saved in %s" % (args.train_epoch, model_dir))
#    print("saving filepath directory: %s" % model_dir)
    saved_model_dir = os.path.join(model_dir, "{epoch:04d}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_dir,
                                            monitor='val_top3_accuracy',
                                            save_weights_only=False,
                                            verbose=1,
                                            save_best_only=True,
                                            mode = 'max',
                                            save_freq='epoch')
    return bert_model, cp_callback

def model_fit(bert_model, train_ds, validation_ds, cp_callback, train_epoch):

    print("start training...")

    for i in range(train_epoch):
            bert_model.fit(
                x=train_ds,
                initial_epoch=i,
                epochs=i + 1,
                validation_data=validation_ds,
                callbacks=[cp_callback])

    print("finish training...")
    return bert_model

## Step 4. Evaluate multiple saved models with the test data.
def model_test(test_ds, args):

    def top1_metric_fn():
      return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 1, name = 'top1_accuracy', dtype=tf.float32)
    
    def top3_metric_fn():
      return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 3, name = 'top3_accuracy', dtype=tf.float32)
    
    def top5_metric_fn():
      return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k = 5, name = 'top5_accuracy', dtype=tf.float32)

    # model_dir_list = tf.io.gfile.listdir(args.model_dir)
    # best_model_ckpt = natsort.natsorted(model_dir_list)[3]
    # print("we have %s emsANN: %s, we choose %s " % (len(model_dir_list), model_dir_list, best_model_ckpt))

#    model_dir_list = tf.io.gfile.listdir(args.model_dir)
#    model_dir_list.sort()

#    best_top3_score = 0.0
#    best_bert_model = None
#    best_bert_idx = -1

    # for idx, ckpt_model_dir in enumerate(model_dir_list):
    # best_model_ckpt = os.path.join(args.model_dir, best_model_ckpt)
    best_model_ckpt = args.test_model_path
    # best_model_ckpt = os.path.join(args.model_dir, ckpt_model_dir)      
#        args.init_model = ckpt_model
#        print("testing the model: %s" % args.init_model)
    test_data_size = len(list(test_ds))
    print("testing the model: %s with size %s" % (best_model_ckpt, test_data_size))
    bert_model = tf.keras.models.load_model(best_model_ckpt, compile=False)
    bert_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[top1_metric_fn(), top3_metric_fn(), top5_metric_fn()])

    # measure the inference latency

    time_s = datetime.now()
    eval_result = bert_model.evaluate(test_ds)
    time_t = datetime.now() - time_s
    time_a = time_t / test_data_size
    print("inference time of model %s on server is %s" % (best_model_ckpt, time_a))

#    print("test result %s" % eval_result)
#    if eval_result[2] > best_top3_score:
#        best_top3_score = eval_result[2]
#        best_bert_model = bert_model
#        best_bert_idx = idx
  
#    best_bert_model = bert_model
#    print("selected best model: %s" % best_bert_idx)

    return bert_model

# Step 5. Export as a TensorFlow Lite model.

def model_save_tflite(bert_model, args):
    def _get_params(f, **kwargs):
      """Gets parameters of the function `f` from `**kwargs`."""
      parameters = inspect.signature(f).parameters
      f_kwargs = {}  # kwargs for the function `f`
      for param_name in parameters.keys():
        if param_name in kwargs:
          f_kwargs[param_name] = kwargs.pop(param_name)
      return f_kwargs, kwargs
    
    logging.info("Converting the tensorflow model to the tflite model")
    
    export_dir = os.path.join(args.eval_dir, "export_tflite")
#    export_dir = 'export_tflite'
    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)
    #with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
    #  f.write(tflite_model)
    
    tflite_filename = args.tflite_name
    tflite_filepath = os.path.join(export_dir, tflite_filename)
    quant_config = quant_configs.QuantizationConfig.for_dynamic()
    quant_config.experimental_new_quantizer = True
    print(bert_model.inputs)

    for model_input in bert_model.inputs:
#        new_shape = [args.test_batch_size] + model_input.shape[1:]
        new_shape = [1] + model_input.shape[1:]
        model_input.set_shape(new_shape)    
        
    print(bert_model.inputs)
    
    #temp_dir_name = tempfile.TemporaryDirectory()
    with tempfile.TemporaryDirectory() as temp_dir_name:
        print(temp_dir_name)
        save_path = os.path.join(temp_dir_name, 'saved_model')
        bert_model.save(save_path, include_optimizer=False, save_format='tf')
        converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS,)
    #    q_cfg = quant_config.QuantizationConfig()
        converter = quant_config.get_converter_with_quantization(converter, preprocess=None)
        converter.target_spec.supported_ops = supported_ops
    
        tflite_model = converter.convert()
    
        with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
          f.write(tflite_model)
        logging.info("TensorFlow Lite model exported successfully: %s" % tflite_filepath)
    
    
    #def build_vocab_tokenizer(model_uri, do_lower_case):
    
    with tempfile.TemporaryDirectory() as temp_dir:                                            
      logging.info('Vocab file and label file are inside the '
                                'TFLite model with metadata.')
      vocab_filepath = os.path.join(temp_dir, 'vocab.txt')
      vocab_file, _ = build_vocab_tokenizer(args.init_model, True)
      tf.io.gfile.copy(vocab_file, vocab_filepath, overwrite=True)
      logging.info('Saved vocabulary in %s.', vocab_filepath)
    #  self.model_spec.save_vocab(vocab_filepath)
      label_filepath = os.path.join(temp_dir, 'labels.txt')
      with tf.io.gfile.GFile(label_filepath, 'w') as f:
        f.write('\n'.join(test_meta_data['index_to_label']))  
    #  self._export_labels(label_filepath)
    
    #  export_dir = os.path.dirname(tflite_filepath)
    #  if isinstance(self.model_spec, text_spec.BertClassifierModelSpec):
      model_info = bert_metadata_writer.ClassifierSpecificInfo(
          name= 'bert text classifier',
          version='v2',
          description=bert_metadata_writer.DEFAULT_DESCRIPTION,
    #      input_names=bert_metadata_writer.bert_qa_inputs(
    #          ids_name=model_spec.tflite_input_name['ids'],
    #          mask_name=model_spec.tflite_input_name['mask'],
    #          segment_ids_name=model_spec.tflite_input_name['segment_ids']),
          input_names=bert_metadata_writer.bert_qa_inputs(
              ids_name='serving_default_input_word_ids:0',
              mask_name='serving_default_input_mask:0',
              segment_ids_name='serving_default_input_type_ids:0'),
          tokenizer_type=bert_metadata_writer.Tokenizer.BERT_TOKENIZER,
          vocab_file=vocab_filepath,
          label_file=label_filepath)
    
    
      
    #  model_info = _get_bert_model_info(self.model_spec, vocab_filepath,
    #                                      label_filepath)
      populator = bert_metadata_writer.MetadataPopulatorForBertTextClassifier(
            tflite_filepath, export_dir, model_info)
    #  elif isinstance(self.model_spec, text_spec.AverageWordVecModelSpec):
    #    model_info = _get_model_info(self.model_spec.name)
    #    populator = metadata_writer.MetadataPopulatorForTextClassifier(
    #        tflite_filepath, export_dir, model_info, label_filepath,
    #        vocab_filepath)
    #  else:
    #    raise ValueError('Model Specification is not supported to writing '
    #                     'metadata into TFLite. Please set '
    #                     '`with_metadata=False` or write metadata by '
    #                     'yourself.')
    #  populator.populate(export_metadata_json_file=False)
      populator.populate(False)

def test_tflite(ds, args):

  def generate_elements(d):
    for element in d.as_numpy_iterator():
      yield element

  ds = ds.unbatch()
  print("dataset size for tflite: %s" % len(list(ds)))

  print("=========== evaluating tflite ==============")
  tflite_filepath = args.test_tflite_model_path
  # tflite_filepath = os.path.join(args.eval_dir, "export_tflite", args.tflite_name)

  with tf.io.gfile.GFile(tflite_filepath, 'rb') as f:
    tflite_model = f.read()
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  y_true, y_pred = [], []
#  inference_latency = []
#  lite_runner = get_lite_runner(tflite_filepath)
  #for i, (feature, label) in enumerate(generate_elements(ds)):
  time_s = datetime.now()
  log_steps = 1000
  for i, (feature, label) in enumerate(list(ds)):
#    print("%s-th batch dataset size for tflite" % i)
    if i % log_steps == 0:
        print("Processing example %s(%s) with tflite" % (i, len(list(ds))))
    #tf.compat.v1.logging.log_every_n(tf.compat.v1.logging.DEBUG,
    #                                 'Processing example: #%d\n%s', log_steps,
    #                                 i, feature)
    #      'input_word_ids': record['input_ids'],
    #  'input_mask': record['input_mask'],
    #  'input_type_ids': record['segment_ids']
    #print(feature)
    input_ids = feature['input_word_ids']
    input_mask = feature['input_mask']
    segment_ids = feature['input_type_ids']

    input_ids = np.array(input_ids, dtype=np.int32)
#    print(input_ids.shape)
    input_mask = np.array(input_mask, dtype=np.int32)
    segment_ids = np.array(segment_ids, dtype=np.int32)

    interpreter.set_tensor(input_details[0]["index"], input_ids)
    interpreter.set_tensor(input_details[2]["index"], input_mask)
    interpreter.set_tensor(input_details[1]["index"], segment_ids)
    interpreter.invoke()

    probabilities = interpreter.get_tensor(output_details[0]["index"])[0]
#    probabilities = lite_runner.run(input_ids, input_mask, segment_ids)
    y_pred.append(probabilities)
    y_true.append(label)

  time_t = datetime.now() - time_s
  time_a = time_t / len(y_pred)
  print("tflite inference time of model %s on server is %s" % (tflite_filepath, time_a))
#  time_t = datetime.now() - time_s
#  print()

  m1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
  m1.update_state(y_true, y_pred)

  m3 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
  m3.update_state(y_true, y_pred)

  m5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
  m5.update_state(y_true, y_pred)

  print("Here is the top1/3/5 using tf sparse topk:")
  print(m1.result().numpy(), m3.result().numpy(), m5.result().numpy())

  topk_prob, topk_indices = tf.math.top_k(y_pred, k=5)
  #topk_prob = topk_prob.numpy()
  #topk_prob = tf.make_ndarray(topk_prob)
  #topk_prob = topk_prob.tolist()
  topk_indices = topk_indices.numpy()
  #topk_indices = tf.make_ndarray(topk_indices)
  topk_indices = topk_indices.tolist()

#  assert(len(topk_indices) == len(example_label_ids))
  total_count = 0
  top1 = 0.000
  top3 = 0.000
  top5 = 0.000
  for (i, topk_index) in enumerate(topk_indices):

      true_label_id = y_true[i]
      true_label_id = true_label_id.numpy()

      if topk_index[0] == true_label_id:
          top1 += 1.0
      if true_label_id in set(topk_index[0:3]):
          top3 += 1.0
      if true_label_id in set(topk_index):
          top5 += 1.0
      total_count += 1

  print("Here is the top1/3/5 using tf math topk:")
  print(top1/total_count, top3/total_count, top5/total_count)



if __name__ == "__main__":
    
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for EMSBert")
    parser.add_argument("--eval_dir", action='store', type=str, default = "eval_pretrain", help="directory containing resources for specific purposes")
    parser.add_argument("--init_model", action='store', type=str, default = "/home/liuyi/tflite_experimental/train_pipeline_standalone/init_models/experts_bert_pubmed_2", help="directory storing the different initialization models")
    parser.add_argument("--cuda_device", action='store', type=str, default = "1", help="indicate the cuda device number")
    parser.add_argument("--max_seq_len", type=int, default=128, help="maximum sequence length")
    parser.add_argument("--train_file", action='store', type=str, default="train.tsv", help="train file name")
    parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--eval_file", action='store', type=str, default="eval.tsv", help="eval file name")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="eval batch size")
    parser.add_argument("--test_file", action='store', type=str, default="test.tsv", help="test file name")
    parser.add_argument("--test_batch_size", type=int, default=64, help="test batch size")
    parser.add_argument("--train_epoch", type=int, default=10, help="epochs for training")

    parser.add_argument("--model_dir", action='store', type=str, default = 'saved_model', help = "indicate where to store the trained models")
    parser.add_argument("--do_train", action='store_true', default=False, help="indicate whether to do training")
    parser.add_argument("--do_test", action='store_true', default=False, help="indicate whether to do testing")
    parser.add_argument("--save_tflite", action='store_true', default=False, help="indicate whether to save tflite models")
    parser.add_argument("--test_tflite", action='store_true', default=False, help="indicate whether to test tflite models")
    parser.add_argument("--tflite_name", action='store', type=str, default='model.tflite', help = "indicate the tflite model name")
    parser.add_argument("--do_predict", action='store_true', default=False, help="indicate whether to do predict")
    parser.add_argument("--bert_trainable", action='store_false', default=True, help="indicate whether we want to freeze the loaded bert model")

    parser.add_argument("--test_model_path", action='store', type=str, help = "indicate where to store the trained models")
    parser.add_argument("--test_tflite_model_path", action='store', type=str, help = "indicate where to store the trained models")    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    refinfo = RefInfo()

    train_ds, train_meta_data, validation_ds, eval_meta_data, test_ds, test_meta_data = prepare_dataset(args, refinfo)
    if args.do_train:
        bert_model, callbacks = prepare_model(train_ds, train_meta_data, validation_ds, args)
        bert_model = model_fit(bert_model, train_ds, validation_ds, callbacks, args.train_epoch)
    if args.do_test:
#        model_test(bert_model, test_ds)
        #best_bert_model = model_test(train_ds, train_meta_data, validation_ds, test_ds, args)
        assert args.test_model_path != None
        best_bert_model = model_test(test_ds, args)
#        print("best_bert_model: ", best_bert_model)
        if args.save_tflite:
            model_save_tflite(best_bert_model, args)

    if args.test_tflite:
        test_tflite(test_ds, args)
    

#    if args.do_predict:
#        model_predict(train_ds, train_meta_data, validation_ds, test_ds, args)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

   
