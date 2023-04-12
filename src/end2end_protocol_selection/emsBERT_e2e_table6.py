import numpy as np
import os
import collections
import csv
from absl import logging
logging.set_verbosity(logging.INFO)
import tensorflow as tf
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
import time

#import TFUtils as tfutil
import pandas as pd
import sys
import match_cloud_transcript_concepts_table6 as sota

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
               text_a,
               label=None):
    self.guid = guid
    self.text_a = text_a
    self.label = label

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

def _load(true_tfrecord_file, transcribed_tfrecord_file, meta_data_file, max_seq_len, test_tflite=False):
  """Loads data from tfrecord file and metada file."""

  name_to_features = get_name_to_features(max_seq_len)

  if test_tflite:
    name_to_features = get_name_to_features_tflite(max_seq_len)

  true_dataset = single_file_dataset(true_tfrecord_file, name_to_features)
  true_dataset = true_dataset.map(select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)
  transcribed_dataset = single_file_dataset(transcribed_tfrecord_file, name_to_features)
  transcribed_dataset = transcribed_dataset.map(select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)

  meta_data = file_util.load_json_file(meta_data_file)

  # logging.info(
  #     'Load preprocessed data and metadata from %s, %s and %s ', true_tfrecord_file, transcribed_tfrecord_file, meta_data_file)
  return true_dataset, transcribed_dataset, meta_data


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  if label_list:
    for (i, label) in enumerate(label_list):
      label_map[label] = i

#  print(example.text_a)
  tokens_a = tokenizer.tokenize(example.text_a)
#  tokens_b = None
#  if example.text_b:
#    tokens_b = tokenizer.tokenize(example.text_b)
#
#  if tokens_b:
#    # Modifies `tokens_a` and `tokens_b` in place so that the total
#    # length is less than the specified length.
#    # Account for [CLS], [SEP], [SEP] with "- 3"
#    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#  else:
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

#  if tokens_b:
#    for token in tokens_b:
#      tokens.append(token)
#      segment_ids.append(1)
#    tokens.append("[SEP]")
#    segment_ids.append(1)

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
#  if ex_index < 5:
#    logging.info("*** Example ***")
#    logging.info("guid: %s", (example.guid))
#    logging.info("tokens: %s",
#                 " ".join([tokenization.printable_text(x) for x in tokens]))
#    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
#    logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
#    logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
#    logging.info("label: %s (id = %s)", example.label, str(label_id))
#    logging.info("int_iden: %s", str(example.int_iden))

  feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
#        int_iden=example.int_iden)

  return feature

def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_file,
                                            label_type=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

#  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
#    if ex_index % 10000 == 0:
#      logging.info("Writing example %d of %d", ex_index, len(examples))

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


def _save_data(true_examples, transcribed_examples, max_seq_len, tokenizer, true_tfrecord_file, transcribed_tfrecord_file, meta_data_file):

  meta_data = file_util.load_json_file(meta_data_file)
  label_names = meta_data["index_to_label"]
#  print(label_names)

  file_based_convert_examples_to_features(true_examples, label_names, max_seq_len, tokenizer, true_tfrecord_file)
  file_based_convert_examples_to_features(transcribed_examples, label_names, max_seq_len, tokenizer, transcribed_tfrecord_file)

def build_vocab_tokenizer(do_lower_case):
  vocab_file = 'vocab.txt'
  assert(tf.io.gfile.exists(vocab_file))

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
  return vocab_file, tokenizer

def from_transcription_result_file(file_path,
                                   label_path,
                                   max_seq_len,
                                   test_tflite=False,
                                   do_lower_case=True,
                                   delimiter='\t',
                                   cache_dir=None):

  meta_data_file = "meta_data.json"

  csv_base_path = file_path.split(".")[0]
  true_tfrecord_file = csv_base_path + "_true.tfrecord"
  transcribed_tfrecord_file = csv_base_path + "_transcribed.tfrecord"
  # print("tfrecord files: %s, %s" % (true_tfrecord_file, transcribed_tfrecord_file))

#  is_cached = (os.path.exists(true_tfrecord_file) and os.path.exists(transcribed_tfrecord_file))
#  if is_cached:
#    return _load(true_tfrecord_file, transcribed_tfrecord_file, meta_data_file, max_seq_len, test_tflite)

  vocab_file, tokenizer = build_vocab_tokenizer(do_lower_case)
  labels = readFile(label_path)
  lines = readFile(file_path)[1:]
  assert len(labels) == len(lines)

  true_examples = []
  transcribed_examples = []
  time_durations = []

  dur_min = sys.float_info.max
  dur_max = sys.float_info.min
  for i, line in enumerate(lines):
      event = line.split("\t")
#      print(event)
      assert len(event) == 4

      time_dur = float(event[1].strip())
      dur_min = min(time_dur, dur_min)
      dur_max = max(time_dur, dur_max)
      time_durations.append(time_dur)

      true_text = event[2]
      transcribed_text = event[3]
#      assert event[4] == ""
      
      label = labels[i]
      guid = '%s-%d' % (csv_base_path, i)
      true_examples.append(InputExample(guid, true_text, label))
      transcribed_examples.append(InputExample(guid, transcribed_text, label))
  # print("duration: min %s, max %s, sum %s, total %s, average %s" % (dur_min, dur_max, sum(time_durations), len(time_durations), sum(time_durations)/len(time_durations)))

  # Saves preprocessed data and other assets into files.
  _save_data(true_examples, transcribed_examples, max_seq_len, tokenizer, true_tfrecord_file, transcribed_tfrecord_file, meta_data_file)

  # Loads data from cache directory.
  return _load(true_tfrecord_file, transcribed_tfrecord_file, meta_data_file, max_seq_len, test_tflite)

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
#  if is_training:
#    if shuffle:
#      buffer_size = 3 * batch_size
#      ds = ds.shuffle(buffer_size=buffer_size)

  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

def top1_metric_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k = 1, name = 'top1_accuracy', dtype=tf.float32)

def top3_metric_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k = 3, name = 'top3_accuracy', dtype=tf.float32)

def top5_metric_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k = 5, name = 'top5_accuracy', dtype=tf.float32)

# the labels for all speakers are all separate
def merge_labels(args, spks):

    labels_all_dict = dict()
    for spk_idx, spk in enumerate(spks[0:-1]):

        finetune_test_path = os.path.join(args.transcription_dir, spk, "finetune_test.tsv")
        test_lines = readFile(finetune_test_path)[1:]
        label_path = os.path.join(args.transcription_dir, spk, "labels.txt")
        labels = readFile(label_path)

        assert len(test_lines) == len(labels)
        for test_line, label in zip(test_lines, labels):
            audio_path = test_line.split("\t")[0]
            labels_all_dict[audio_path] = label

    return labels_all_dict

def get_label_path_for_all(strategy_path, test_out_path, labels_all_dict):

    labels_all = []
    lines = readFile(test_out_path)[1:]
    assert len(lines) == 600
    
    for line_idx, line in enumerate(lines):
        audio_path = line.split("\t")[0]
        label = labels_all_dict[audio_path]
        labels_all.append(label)
    labels_all_path = os.path.join(strategy_path, "labels.txt")
    writeListFile(labels_all_path, labels_all)

    return labels_all_path

def evaluate_end_to_end(args):

    spks = [
        "audio_tian",
        "audio_liuyi",
        "audio_yichen",
        "audio_radu",
        "audio_amran",
        "audio_michael",
        "audio_all",
    ]
    labels_all_dict = merge_labels(args, spks)
    # print("labels_all_dict len: %s" % len(labels_all_dict))

    speech_models = [
        "conformer",
        # "contextNet",
        # "rnnt"
    ]

    training_strategy = [
        # "PretrainLibrispeech_DirectInferenceEMS",
        "PretrainLibrispeech_TrainEMS",
        # "TrainFromScratchEMS"
    ]

    true_result_list = []
    transcribed_result_list = []
    true_tflite_result_list = []
    transcribed_tflite_result_list = []

    for spk_idx, spk in enumerate(spks):
        for model_idx, speech_model in enumerate(speech_models):
            for strategy_idx, strategy in enumerate(training_strategy):
                model_training_strategy = speech_model + "_" + strategy
                strategy_path = os.path.join(args.transcription_dir, spk, speech_model, model_training_strategy)
                assert os.path.isdir(strategy_path)

                # if spk_idx != 6:
                #     continue


                # if (spk == "audio_radu" and speech_model == "conformer" and strategy == "TrainFromScratchEMS"):
                #     print("skipping %s" % strategy_path)
                #     continue
                # if (spk == "audio_michael" and speech_model == "conformer" and strategy == "TrainFromScratchEMS"):
                #     print("skipping %s" % strategy_path)
                #     continue


                # if strategy_idx == 0:
                #     test_out_path = os.path.join(strategy_path, "test.output")
                # else:
                test_out_path = os.path.join(strategy_path, "test_output.tsv")
        
                if spk_idx != 6:
                    label_path = os.path.join(args.transcription_dir, spk, "labels.txt")
                else:
                    label_path = get_label_path_for_all(strategy_path, test_out_path, labels_all_dict)
                
                # print("strategy_path: %s" % strategy_path)
                # print("label_path: %s" % label_path)
                # print("test_out_path: %s" % test_out_path)                

                true_dataset, transcribed_dataset, meta_data = from_transcription_result_file(test_out_path, label_path, args.max_seq_len)
                true_ds = gen_dataset(true_dataset, args.batch_size)
                transcribed_ds = gen_dataset(transcribed_dataset, args.batch_size)

                model_path = args.protocol_model
                print("we are using %s for end2end evaluation" % model_path)

                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                        optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=[top1_metric_fn(), top3_metric_fn(), top5_metric_fn()])
            
                # print("true text accuracy:")
                true_result = model.evaluate(true_ds)
                # print("transcribed text accuracy:")
                transcribed_result = model.evaluate(transcribed_ds)

                true_dataset_tflite, transcribed_dataset_tflite, meta_data = from_transcription_result_file(test_out_path, label_path, args.max_seq_len, test_tflite = True)
                # print("true text tflite accuracy:")
#                true_ds_tflite = gen_dataset(true_dataset)
                true_tflite_result = test_tflite(true_dataset_tflite, args)
                # print("transcribed text tflite accuracy:")
#                transcribed_dataset_tflite = gen_dataset(transcribed_dataset)
                transcribed_tflite_result = test_tflite(transcribed_dataset_tflite, args)

                # sota_result = sota.sota_concept_matching(args)
                # print("\n##### SOTA E2E Protocol Selection Accuracy #####")                
                # print("Truth: Server %s, PH-1 %s" % ([round(sota_result[0][0],2), round(sota_result[0][1],2), round(sota_result[0][2],2)], 
                #                                     [round(sota_result[0][3],2), round(sota_result[0][4],2), round(sota_result[0][5],2)]))

                # for idx, result in enumerate(sota_result[1:]):
                #     print("GC%d: Server %s, PH-1 %s" % (idx+1, [round(result[0],2), round(result[1],2), round(result[2],2)], 
                #                                     [round(result[3],2), round(result[4],2), round(result[5],2)]))

                true_result_list.append(true_result)
                transcribed_result_list.append(transcribed_result)
                true_tflite_result_list.append(true_tflite_result)
                transcribed_tflite_result_list.append(transcribed_tflite_result)

    sota_result = sota.sota_concept_matching(args)
    # print("sota_result: %s x %s x%s" % (len(sota_result), len(sota_result[0]), len(sota_result[0][0])))
    # for idx, result in enumerate(sota_result):
    #     print("Truth %s \t %s" % ([round(result[0][0], 2), round(result[0][1], 2), round(result[0][2], 2)],
    #                               [round(result[0][3], 2), round(result[0][4], 2), round(result[0][5], 2)]))
    #     print("E2E %s \t %s" % ([round(result[7][0], 2), round(result[7][1], 2), round(result[7][2], 2)],
    #                               [round(result[7][3], 2), round(result[7][4], 2), round(result[7][5], 2)]))

    print(len(true_result_list))

    print("\n##### EMSMobileBERT E2E Protocol Selection Accuracy #####")
    print("\t\t SOTA (GC7)  \t\t\t\t\t EMSAssist (ours)")
    print("\t\t Server \t\t PH-1\t\t\t Server \t\t PH-1")    
    for idx, line in enumerate(true_result_list[:6]):

        result = sota_result[idx]

        print("\t Truth \t %s \t %s \t %s \t %s" % ( [round(result[0][0], 2), round(result[0][1], 2), round(result[0][2], 2)],
                                [round(result[0][3], 2), round(result[0][4], 2), round(result[0][5], 2)],
                                [round(line[1], 2), round(line[2], 2), round(line[3], 2)], 
                                [round(true_tflite_result_list[idx][0], 2), round(true_tflite_result_list[idx][1], 2), round(true_tflite_result_list[idx][2], 2)]))
        print("\t E2E \t %s \t %s \t %s \t %s" % ([round(result[7][0], 2), round(result[7][1], 2), round(result[7][2], 2)],
                                [round(result[7][3], 2), round(result[7][4], 2), round(result[7][5], 2)],
                                [round(transcribed_result_list[idx][1], 2), round(transcribed_result_list[idx][2], 2), round(transcribed_result_list[idx][3], 2)], 
                                [round(transcribed_tflite_result_list[idx][0],2), round(transcribed_tflite_result_list[idx][2],2), round(transcribed_tflite_result_list[idx][2],2)]))

                # print("Truth: Server %s, PH-1 %s" % ([round(true_result[1],2), round(true_result[2],2), round(true_result[3],2)], 
                #                                     [round(true_tflite_result[0],2), round(true_tflite_result[1],2), round(true_tflite_result[2],2)]))
                # print("E2E: Server %s, PH-1 %s" % ([round(transcribed_result[1],2), round(transcribed_result[2],2), round(transcribed_result[3],2)], 
                #                                     [round(transcribed_tflite_result[0],2), round(transcribed_tflite_result[1],2), round(transcribed_tflite_result[2],2)]))


def test_tflite(ds, args):

  def generate_elements(d):
    for element in d.as_numpy_iterator():
      yield element

#  ds = ds.unbatch()
  assert len(list(ds)) == 100 or len(list(ds)) == 600
#  assert len(list(ds)) == 100
#  print("dataset size for tflite: %s" % len(list(ds)))

#  print("=========== evaluating tflite ==============")
#  tflite_filepath = os.path.join(args.eval_dir, "export_tflite", args.tflite_name)
  tflite_filepath = args.protocol_tflite_model

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
#    if i % log_steps == 0:
#        print("Processing example %s(%s) with tflite" % (i, len(list(ds))))
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

  # print("Here is the top1/3/5 using tf sparse topk:")
  return [m1.result().numpy(), m3.result().numpy(), m5.result().numpy()]

if __name__ == "__main__":
    
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the e2e eval functions for EMSBert")
    parser.add_argument("--transcription_dir", action='store', type=str, default = "/home/liuyi/emsAssist_mobisys22/data/transcription_text", help="directory containing all transcription results")
    parser.add_argument("--protocol_model", action='store', type=str, required=True)
    parser.add_argument("--protocol_tflite_model", action='store', type=str, required=True)
    parser.add_argument("--cuda_device", action='store', type=str, default = "1", help="indicate the cuda device number")
    parser.add_argument("--max_seq_len", type=int, default=128, help="maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--regen", action='store_true', default=False, help="indicate whether to regenerate files for concept matching")
    parser.add_argument("--concept_set_source", action='store', type=str, default="both", help="protocol, nemsis, both")
    parser.add_argument("--metric", action='store', type=str, default="cosine", help="cosine, dot")
    parser.add_argument("--main_dir", action='store', type=str, default="google_cloud")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    #est_ds, test_meta_data = prepare_dataset(args)
    evaluate_end_to_end(args)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

   
