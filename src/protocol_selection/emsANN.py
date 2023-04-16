import collections
import argparse
import tensorflow as tf
import numpy as np
#from tensorflow import keras
from datetime import datetime
import file_util
import tokenization
import optimization
import os
import csv
import random
from absl import logging
import natsort
import tempfile
logging.set_verbosity(logging.INFO)
import quantization_configs as quant_configs

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
               feature_type,
               input_ids,
               label_id):
    self.feature_type = feature_type
    self.input_ids = input_ids
    self.label_id = label_id

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
 
def counting_dict(list_to_count: list):

  count_d = collections.OrderedDict()
#  print(type(count_d))
  for e in list_to_count:
    if e in count_d:
      count_d[e] += 1
    else:
      count_d[e] = 1
  return count_d


def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
#    if t.dtype == tf.int64:
#      t = tf.cast(t, tf.int32)
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
      'label_id': tf.io.FixedLenFeature([], tf.int64),
  }
  return name_to_features

def get_name_to_features_tflite(seq_len):

#  print("test_tflite true ")
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([1, seq_len], tf.int64),
      'label_id': tf.io.FixedLenFeature([], tf.int64),
  }
  return name_to_features

def select_data_from_record(record):
  """Dispatches records to features and labels."""
  x = record['input_ids']
  y = record['label_id']
  return (x, y)

def _load(tfrecord_file, meta_data, test_tflite=False):
  """Loads data from tfrecord file and metada file."""

  feature_dim = meta_data['input_feature_dim']
  name_to_features = get_name_to_features(feature_dim)

  if test_tflite:
    name_to_features = get_name_to_features_tflite(feature_dim)

  dataset = single_file_dataset(tfrecord_file, name_to_features)
  dataset = dataset.map(select_data_from_record, num_parallel_calls=tf.data.AUTOTUNE)

  #meta_data = file_util.load_json_file(meta_data_file)

  #logging.info(
  #    'Load preprocessed data and metadata from %s and %s '
  #    'with size: %d', tfrecord_file, meta_data_file, meta_data['size'])
  return dataset, meta_data

def convert_examples_to_features(examples,
                                 label_list,
                                 tokenizer,
                                 feature_type,
                                 refinfo,
                                 meta_data):
  """Convert a set of `InputExample`s to a TFRecord file."""

  assert feature_type == "token_counts" or feature_type == "tokens" or feature_type == "word_counts" or feature_type == "words" or feature_type == "code_counts" or feature_type == "codes"

  def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f
  features = []

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  
  logging.info("writing examples tfrecord to %s" % meta_data['tfrecord_file_path'])
  writer = tf.io.TFRecordWriter(meta_data['tfrecord_file_path'])
  if feature_type == "token_counts" or feature_type == "tokens":

    #for i, feature in enumerate(features):
    for (ex_index, example) in enumerate(examples):
      #if ex_index % 10000 == 0:
      #  logging.info("converting example %d of %d", ex_index, len(examples))

      tokens = tokenizer.tokenize(example.text)
      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      label_id = label_map[example.label]
      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("tokens: %s",
                     " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])

      one_hot_enc = tf.one_hot(indices=input_ids, depth=len(tokenizer.vocab), dtype=tf.int64)
      input_ids = tf.math.reduce_max(one_hot_enc, axis=0)
      feature = collections.OrderedDict()
      feature["input_ids"] = create_int_feature(input_ids.numpy())
      feature["label_id"] = create_int_feature([label_id])
      tf_feature = tf.train.Example(features=tf.train.Features(feature=feature))
      if ex_index % 10000 == 0:
        logging.info("Writing example %d of %d", ex_index, len(examples))
      writer.write(tf_feature.SerializeToString())
    writer.close()
    #features.append(tf_feature)
    #return features

  elif feature_type == "word_counts" or feature_type == "words":
#    logging.info("writing examples tfrecord to %s" % meta_data['tfrecord_file_path'])
#    writer = tf.io.TFRecordWriter(meta_data['tfrecord_file_path'])
    for (ex_index, example) in enumerate(examples):
#      if ex_index % 10000 == 0:
#        logging.info("converting example %d of %d", ex_index, len(examples))

      col_index_cnt = counting_dict(example.words)
      input_ids = [refinfo.word_map[word] for word, cnt in col_index_cnt.items()]
      label_id = label_map[example.label]
      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("words: %s", ", ".join([str(x) for x in example.words]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])

      one_hot_enc = tf.one_hot(indices=input_ids, depth=len(refinfo.word_map), dtype=tf.int64)
      input_ids = tf.math.reduce_max(one_hot_enc, axis=0)
      feature = collections.OrderedDict()
      feature["input_ids"] = create_int_feature(input_ids.numpy())
      feature["label_id"] = create_int_feature([label_id])
      tf_feature = tf.train.Example(features=tf.train.Features(feature=feature))
      if ex_index % 10000 == 0:
        logging.info("Writing example %d of %d", ex_index, len(examples))
      writer.write(tf_feature.SerializeToString())
    writer.close()
#      features.append(tf_feature)
#    return features
      
  elif feature_type == "code_counts" or feature_type == "codes":
#    logging.info("writing examples tfrecord to %s" % meta_data['tfrecord_file_path'])
#    writer = tf.io.TFRecordWriter(meta_data['tfrecord_file_path'])
    for (ex_index, example) in enumerate(examples):
#      if ex_index % 10000 == 0:
#        logging.info("converting example %d of %d", ex_index, len(examples))

      col_index_cnt = counting_dict(example.codes)
      input_ids = [refinfo.code_map[code] for code, cnt in col_index_cnt.items()]
      label_id = label_map[example.label]
      if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s", (example.guid))
        logging.info("text: %s", example.text)
        logging.info("codes: %s", ", ".join([str(x) for x in example.codes]))
        logging.info("label: %s (id = %s)", example.label, label_map[example.label])

      one_hot_enc = tf.one_hot(indices=input_ids, depth=len(refinfo.code_map), dtype=tf.int64)
      input_ids = tf.math.reduce_max(one_hot_enc, axis=0)
      feature = collections.OrderedDict()
      feature["input_ids"] = create_int_feature(input_ids.numpy())
      feature["label_id"] = create_int_feature([label_id])
      tf_feature = tf.train.Example(features=tf.train.Features(feature=feature))
      if ex_index % 10000 == 0:
        logging.info("Writing example %d of %d", ex_index, len(examples))
      writer.write(tf_feature.SerializeToString())
    writer.close()
#      features.append(tf_feature)
#
#    return features

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

def from_codes(filename,
             text_column,
             label_column,
             do_lower_case,
             shuffle,
             vocab_file,
             feature_type,
             is_training,
             refinfo,
             test_tflite=False,
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
  
#  codes_map, words_map = codes_words_map()

#  random.seed(global_seed)
#  if shuffle:    
#    random.shuffle(text_lines)
#    random.shuffle(word_lines)
#    random.shuffle(code_lines)

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
    words, word_label = word_line[0], word_line[1]
    codes, code_label = code_line[0], code_line[1]
    assert label == word_label
    assert label == code_label
    guid = '%s-%d' % (csv_name, i)
    examples.append(InputExample(guid, text, words, codes, label))

  meta_data = {
      'size': len(examples),
      'num_classes': len(label_names),
      'index_to_label': label_names,
  }

  if feature_type == "tokens" or feature_type == "token_counts":
    meta_data['input_feature_dim'] = len(tokenizer.vocab)
  elif feature_type == "words" or feature_type == "word_counts":
    meta_data['input_feature_dim'] = len(refinfo.word_map)
  elif feature_type == "codes" or feature_type == "code_counts":
    meta_data['input_feature_dim'] = len(refinfo.code_map)

  tfrecord_file = csv_name.split('.')[0] + ".tfrecord"
  tfrecord_file_path = os.path.join(args.eval_dir, tfrecord_file)
  # if tf.io.gfile.exists(tfrecord_file_path):
  #   return _load(tfrecord_file_path, meta_data, test_tflite)

  random.seed(global_seed)
  if shuffle:    
    random.shuffle(examples)

  meta_data['tfrecord_file_path'] = tfrecord_file_path
  #features = convert_examples_to_features(examples, label_names, tokenizer, feature_type, refinfo)
  convert_examples_to_features(examples, label_names, tokenizer, feature_type, refinfo, meta_data)

#  logging.info("writing examples tfrecord to %s" % tfrecord_file_path)
#  writer = tf.io.TFRecordWriter(tfrecord_file_path)
#  for i, feature in enumerate(features):
#    if i % 10000 == 0:
#      logging.info("Writing example %d of %d", i, len(features))
#    writer.write(feature.SerializeToString())
#  writer.close()

  return _load(tfrecord_file_path, meta_data, test_tflite)

def prepare_dataset(args, refinfo):

    test_file_name = os.path.join(args.eval_dir, args.test_file)
    print("test_file_path: %s" % test_file_name)
#    test_x, test_y, test_meta_data = from_codes(
    test_ds, test_meta_data = from_codes(
          filename=test_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=False,
          shuffle=False,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo,
          test_tflite=args.test_tflite)
    test_ds = gen_dataset(test_ds, args.test_batch_size, is_training=False)
    print(test_meta_data)

    if args.test_only or args.test_tflite:
        return None, None, None, None, test_ds, test_meta_data

    train_file_name = os.path.join(args.eval_dir, args.train_file)
    train_ds, train_meta_data = from_codes(
#    train_x, train_y, train_meta_data = from_codes(
          filename=train_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=True,
          shuffle=True,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo,
          test_tflite=args.test_tflite)
    train_ds = gen_dataset(train_ds, args.train_batch_size, is_training=True)
    print(train_meta_data)

    eval_file_name = os.path.join(args.eval_dir, args.eval_file)
#    eval_x, eval_y, eval_meta_data = from_codes(
    eval_ds, eval_meta_data = from_codes(
          filename=eval_file_name,
          text_column='ps_pi_as_si_desc_c_mml_c',
          label_column='label',
          do_lower_case=True,
          delimiter='\t',
          is_training=False,
          shuffle=False,
          vocab_file=args.vocab_file,
          feature_type=args.feature_type,
          refinfo=refinfo,
          test_tflite=args.test_tflite)
    eval_ds = gen_dataset(eval_ds, args.eval_batch_size, is_training=False)
    print(eval_meta_data)

    return train_ds, train_meta_data, eval_ds, eval_meta_data, test_ds, test_meta_data

def top1_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 1, name = 'top1', dtype=tf.float32)

def top3_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 3, name = 'top3', dtype=tf.float32)

def top5_acc_fn():
  return tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k = 5, name = 'top5', dtype=tf.float32)

def train_ann_model(prepared_data, args):

#    train_x, train_y, train_meta_data, eval_x, eval_y, eval_meta_data, test_x, test_y, test_meta_data = prepared_data 
    train_ds, train_meta_data, eval_ds, eval_meta_data, test_ds, test_meta_data = prepared_data

#    train_ds = test_ds
#    train_meta_data = test_meta_data
#    eval_ds = test_ds
#    eval_meta_data = test_meta_data

    initializer = tf.keras.initializers.TruncatedNormal(seed=global_seed)
    feature_dim = train_meta_data['input_feature_dim']
#    train_batch_size = args.train_batch_size
#    eval_batch_size = args.eval_batch_size
#    test_batch_size = args.test_batch_size
#    train_epoch = args.train_epoch
    num_classes = train_meta_data['num_classes']

    saved_model_dir = os.path.join(args.model_dir, "{epoch:04d}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_dir,
                                            monitor='val_top3',
                                            mode = 'max',
                                            save_weights_only=False,
                                            verbose=1,
                                            save_best_only=True,
                                            save_freq='epoch')    

#    inputs = keras.layers.InputLayer(input_shape=(feature_dim,),name="ANNInput")
    inputs = tf.keras.Input(shape=(feature_dim,), name="ANNInput", sparse=False, dtype=tf.int64)
    print("input shape: %s" % inputs.shape)
    layer1 = tf.keras.layers.Dense(
            1024,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            name="layer1",
            activation="softmax",
            dtype=tf.float32)(inputs)
    print("layer1 shape: %s" % layer1.shape)
    layer1 = tf.keras.layers.BatchNormalization()(layer1)

    layer2 = tf.keras.layers.Dense(
            512,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            name="layer2",
            activation="softmax",
            dtype=tf.float32)(layer1)
    print("layer2 shape: %s" % layer2.shape)
    layer2 = tf.keras.layers.BatchNormalization()(layer2)

    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        name="output",
        activation="softmax",
        dtype=tf.float32)(layer2)

    classifier = tf.keras.Model(inputs = inputs, outputs = outputs)
    classifier.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[top1_acc_fn(), top3_acc_fn(), top5_acc_fn()],
    )    

    history = classifier.fit(
        x = train_ds,
#        y = train_y,
        batch_size = args.train_batch_size,
#        validation_data = (eval_x, eval_y),
        validation_data = eval_ds,
        epochs = args.train_epoch,
        callbacks=[cp_callback]
    )

    test_ann_model(prepared_data, args)

def model_save_tflite(model, args):
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
    if not tf.io.gfile.exists(export_dir):
      tf.io.gfile.makedirs(export_dir)
    
    tflite_filename = args.tflite_name
    tflite_filepath = os.path.join(export_dir, tflite_filename)
    quant_config = quant_configs.QuantizationConfig.for_dynamic()
    quant_config.experimental_new_quantizer = True
    print(model.inputs)

    for model_input in model.inputs:
        new_shape = [1] + model_input.shape[1:]
        model_input.set_shape(new_shape)    
        
    print(model.inputs)
    
    with tempfile.TemporaryDirectory() as temp_dir_name:
        print(temp_dir_name)
        save_path = os.path.join(temp_dir_name, 'saved_model')
        model.save(save_path, include_optimizer=False, save_format='tf')
        converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS)
        converter = quant_config.get_converter_with_quantization(converter, preprocess=None)
        converter.target_spec.supported_ops = supported_ops
    
        tflite_model = converter.convert()
    
        with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
          f.write(tflite_model)
        logging.info("TensorFlow Lite model exported successfully: %s" % tflite_filepath)
    
#    with tempfile.TemporaryDirectory() as temp_dir:                                            
#      logging.info('Vocab file and label file are inside the '
#                                'TFLite model with metadata.')
#      vocab_filepath = os.path.join(temp_dir, 'vocab.txt')
#      vocab_file, _ = build_vocab_tokenizer(args.init_model, True)
#      tf.io.gfile.copy(vocab_file, vocab_filepath, overwrite=True)
#      logging.info('Saved vocabulary in %s.', vocab_filepath)
#      label_filepath = os.path.join(temp_dir, 'labels.txt')
#      with tf.io.gfile.GFile(label_filepath, 'w') as f:
#        f.write('\n'.join(test_meta_data['index_to_label']))  
#      model_info = bert_metadata_writer.ClassifierSpecificInfo(
#          name= 'bert text classifier',
#          version='v2',
#          description=bert_metadata_writer.DEFAULT_DESCRIPTION,
#          input_names=bert_metadata_writer.bert_qa_inputs(
#              ids_name='serving_default_input_word_ids:0',
#              mask_name='serving_default_input_mask:0',
#              segment_ids_name='serving_default_input_type_ids:0'),
#          tokenizer_type=bert_metadata_writer.Tokenizer.BERT_TOKENIZER,
#          vocab_file=vocab_filepath,
#          label_file=label_filepath)
#      populator = bert_metadata_writer.MetadataPopulatorForBertTextClassifier(
#            tflite_filepath, export_dir, model_info)
#      populator.populate(False)
#    evaluate_tflite(args)
    
    
   
def test_ann_model(prepared_data, args):

    train_ds, train_meta_data, eval_ds, eval_meta_data, test_ds, test_meta_data = prepared_data 


    # model_dir_list = tf.io.gfile.listdir(args.model_dir)
    # best_model_ckpt = natsort.natsorted(model_dir_list)[-1]
    # print("we have %s emsANN: %s, we choose %s " % (len(model_dir_list), model_dir_list, best_model_ckpt))
#    print("from saved emsANN, we chose %s" % str(best_model_ckpt[-1]))

    ckpt_model = args.test_model_path

    # ckpt_model = os.path.join(args.model_dir, best_model_ckpt)
    classifier = tf.keras.models.load_model(ckpt_model, compile=False)
    classifier.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[top1_acc_fn(), top3_acc_fn(), top5_acc_fn()])

#    print("\nEvaluate on val data:")
#    classifier.evaluate(eval_ds, batch_size=args.eval_batch_size)



    print("\nEvaluate on test data:")

    time_s = datetime.now()
    classifier.evaluate(test_ds, batch_size=args.test_batch_size)
    time_t = datetime.now() - time_s
    time_a = time_t / len(list(test_ds.unbatch()))
    print("inference time of model %s on server is %s" % (ckpt_model, time_a))

#     print("\n predict test data:")
#     test_Y_pred = classifier.predict(test_ds)
    
#     test_y = []
#     for i, (feature, label) in enumerate(list(test_ds.unbatch())):
#         test_y.append(label)

# #    print("test_y %s" % test_y)

#     # calculate top1, top3, tyop5 accuracy
#     top1_acc = 0
#     top3_acc = 0
#     top5_acc = 0
#     for i in range(len(test_y)):
        
#         if test_y[i] == np.argmax(test_Y_pred[i]):
#             top1_acc += 1
#         if test_y[i] in np.argsort(test_Y_pred[i])[-3:]:
#             top3_acc += 1
#         if test_y[i] in np.argsort(test_Y_pred[i])[-5:]:
#             top5_acc += 1
#     print("top1 testaccuracy: %s" % (top1_acc / len(test_y)))
#     print("top3 testaccuracy: %s" % (top3_acc / len(test_y)))
#     print("top5 test accuracy: %s" % (top5_acc / len(test_y)))

#     model_save_tflite(classifier, args)

#    test_tflite(test_ds, args)

def test_tflite(prepared_data, args):

  train_ds, train_meta_data, eval_ds, eval_meta_data, test_ds, test_meta_data = prepared_data 

  def generate_elements(d):
    for element in d.as_numpy_iterator():
      yield element

  ds = test_ds.unbatch()
  print("dataset size for tflite: %s" % len(list(ds)))

  print("=========== evaluating tflite ==============")
  tflite_filepath = os.path.join(args.eval_dir, "export_tflite", args.tflite_name)

  with tf.io.gfile.GFile(tflite_filepath, 'rb') as f:
    tflite_model = f.read()
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()

  print("input details: ", input_details)

  output_details = interpreter.get_output_details()

  y_true, y_pred = [], []
  time_s = datetime.now()
  log_steps = 1000
  for i, (feature, label) in enumerate(list(ds)):
#    print("%s-th batch dataset size for tflite" % i)
    if i % log_steps == 0:
        print("Processing example %s(%s) with tflite" % (i, len(list(ds))))

#    print(feature)

    #input_ids = feature['input_ids']
#    input_ids = tf.cast(feature, tf.float32)
    input_ids = feature
#    input_ids = feature.numpy()
#    input_ids = np.array(input_ids, dtype=np.int64)
    interpreter.set_tensor(input_details[0]["index"], input_ids)
    interpreter.invoke()

    probabilities = interpreter.get_tensor(output_details[0]["index"])[0]

    y_pred.append(probabilities)
    y_true.append(label)

  time_t = datetime.now() - time_s
  time_a = time_t / len(y_pred)
  print("tflite inference time of model %s on server is %s" % (tflite_filepath, time_a))

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
    parser.add_argument("--eval_dir", action='store', type=str, default = "data", help="directory containing resources for specific purposes")
    parser.add_argument("--train_file", action='store', type=str, default="train.tsv", help="train file name")
    parser.add_argument("--eval_file", action='store', type=str, default="eval.tsv", help="eval file name")
    parser.add_argument("--test_file", action='store', type=str, default="test.tsv", help="test file name")
    parser.add_argument("--vocab_file", action='store', type=str, help="test file name", required=True)
    parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="eval batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="test batch size")
    parser.add_argument("--cuda_device", action='store', type=str, default = "1", help="indicate the cuda device number")
    parser.add_argument("--feature_type", action='store', type=str, default = "tokens", help="indicate the feature type")
    parser.add_argument("--model_dir", action='store', type=str, default = 'saved_model', help = "indicate where to store the trained models")
    parser.add_argument("--test_only", action='store_true', default=False, help="indicate whether to only do testing")
    parser.add_argument("--train_epoch", type=int, default=10, help="epochs for training")
    parser.add_argument("--test_tflite", action='store_true', default=False, help="indicate whether to test tflite models")
    parser.add_argument("--tflite_name", action='store', type=str, default='model.tflite', help = "indicate the tflite model name")
    # parser.add_argument("--cache_dir", action='store', type=str, default = "/slot1/tfrecord_files", help="directory containing resources for specific purposes")

    parser.add_argument("--test_model_path", action='store', type=str, help = "indicate where to store the trained models", required = True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    refinfo = RefInfo()

    prepared_data = prepare_dataset(args, refinfo)

    if args.test_tflite:
        test_tflite(prepared_data, args)
    elif args.test_only:
        test_ann_model(prepared_data, args)
    else:
        train_ann_model(prepared_data, args)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)
