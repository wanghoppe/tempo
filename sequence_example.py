import tensorflow as tf
from magenta.common.sequence_example_lib import count_records, _shuffle_inputs

#use magenta convention
QUEUE_CAPACITY = 500
SHUFFLE_MIN_AFTER_DEQUEUE = QUEUE_CAPACITY // 5

def make_sequence_example(inputs, labels):
    """Returns a SequenceExample for the given inputs and labels.

    Args:
    inputs: A list of input vectors. Each input vector is a list of floats.
    labels: A list of directory mapping keys to int.

    Returns:
    A tf.train.SequenceExample containing inputs and labels.
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_duration = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label['duration']]))
        for label in labels]
    label_silence = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label['silence']]))
        for label in labels]
    label_velocity = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label['velocity']]))
        for label in labels]

    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'duration': tf.train.FeatureList(feature=label_duration),
        'silence': tf.train.FeatureList(feature=label_silence),
        'velocity': tf.train.FeatureList(feature=label_velocity),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def get_padded_batch(file_list, batch_size = 32,
                    input_size = None, shuffle = True):
    """(My inplementation, not used in training)
    Reads batches of SequenceExamples from TFRecords and pads them.

    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.

    Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
    will have a shape [batch_size, num_steps, input_size].
    shuffle: Whether to shuffle the batches.

    Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32.
    duration: A tensor of shape [batch_size, 65] of int64s.
    silence: A tensor of shape [batch_size, 33] of int64s.
    velocity: A tensor of shape [batch_size, 15] of int64s.
    """

    def _parse_function(example_proto):
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[82],dtype=tf.float32),
            'duration': tf.FixedLenSequenceFeature(shape=[],dtype=tf.int64),
            'silence': tf.FixedLenSequenceFeature(shape=[],dtype=tf.int64),
            'velocity': tf.FixedLenSequenceFeature(shape=[],dtype=tf.int64)
        }

        _, parsed_features = \
            tf.parse_single_sequence_example(example_proto,
            sequence_features = sequence_features)
        return parsed_features

    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_function)

    if shuffle: # when training
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)

    dataset = dataset.padded_batch(batch_size, {'inputs':[None,82],
                                'duration':[None],
                                'silence':[None],
                                'velocity':[None]})

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    return (next_element['inputs'],
            next_element['duration'],
            next_element['silence'],
            next_element['velocity'])



# TODO
def get_padded_batch_using_quene(file_list, batch_size = 32, input_size = None,
                                shuffle = True, mode = 'training'):
    """（Using Magenta inplementation)
    (from magenta.common.sequence_example_lib)
    Reads batches of SequenceExamples from TFRecords and pads them.

    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.

    Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
    will have a shape [batch_size, num_steps, input_size].
    num_enqueuing_threads: The number of threads to use for enqueuing
    SequenceExamples.
    shuffle: Whether to shuffle the batches.

    Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32.
    duration: A tensor of shape [batch_size, 65] of int64s.
    silence: A tensor of shape [batch_size, 33] of int64s.
    velocity: A tensor of shape [batch_size, 15] of int64s.
    length: A tensor of shape [batch_size] of int32s. The lengths of each
    SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                            dtype=tf.float32),
        'duration': tf.FixedLenSequenceFeature(shape=[],
                                            dtype=tf.int64),
        'silence': tf.FixedLenSequenceFeature(shape=[],
                                            dtype=tf.int64),
        'velocity': tf.FixedLenSequenceFeature(shape=[],
                                            dtype=tf.int64)
        }

    _, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    length = tf.shape(sequence['inputs'])[0]

    input_tensors = [sequence['inputs'],
                    sequence['duration'],
                    sequence['silence'],
                    sequence['velocity'],
                    length]

    if shuffle:
        min_after_dequeue = count_records(
            file_list, stop_at=SHUFFLE_MIN_AFTER_DEQUEUE)
        input_tensors = _shuffle_inputs(
            input_tensors, capacity=QUEUE_CAPACITY,
            min_after_dequeue=min_after_dequeue,
            num_threads=2)

    tf.logging.info(input_tensors)
    return tf.train.batch(input_tensors,
                        batch_size=batch_size,
                        capacity=QUEUE_CAPACITY,
                        num_threads=2,
                        dynamic_pad=True,
                        allow_smaller_final_batch=False)
