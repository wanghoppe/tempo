import tensorflow as tf
from sequence_example import get_padded_batch, get_padded_batch_using_quene
from magenta.common.sequence_example_lib import flatten_maybe_padded_sequences
import six
from encoder_decoder import MelodyEncoderDecoder

from melody_lib import VELOCITY_VALUE
from melody_lib import MAX_SILENCE, MIN_SILENCE, MAX_DURATION, MIN_DURATION
from melody_lib import DURATION_RANGE, SILENCE_RANGE, VELOCITY_RANGE

from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import session_run_hook
#Some hparams

RNN_LAYERS = [256,256,256]
BATCH_SIZE = 32
NUM_STEP_FOR_TRAINING = 100
TRAINING_DATA = ['training_data/training_melodies.tfrecord']
TRAINING_DIR = 'training'
EVAL_DIR = 'eval'

class LossTooBigHook(session_run_hook.SessionRunHook):
    """Monitors the loss tensor and stops training if loss is too build_graph_fn.

    Can either fail with exception or just stop training.
    """

    def __init__(self, loss_tensor, threshold, fail_on_big_loss=True):
        """Initializes a `LossTooBigHook`.

        Args:
        loss_tensor: `Tensor`, the loss tensor.
        threshold: threshold of the loss.
        fail_on_big_loss: `bool`, whether to raise exception when loss is too big.
        """
        self._loss_tensor = loss_tensor
        self._threshold = threshold
        self._fail_on_big_loss = fail_on_big_loss

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if run_values.results > self._threshold:
            failure_message = \
                "Loss exceed threshold: {0}>{1}.".format(run_values.results,
                                                        self._threshold)
            if self._fail_on_big_loss:
                tf.logging.error(failure_message)
                raise RuntimeError(failure_message)
            else:
                tf.logging.warning(failure_message)
                # We don't raise an error but we request stop without an exception.
                run_context.request_stop()

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.LSTMCell):
    """Makes a RNN cell from the given hyperparameters.

    Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

    Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
    cells = []
    for num_units in rnn_layer_sizes:
        cell = base_cell(num_units)
        if attn_length and not cells:
          # Add attention wrapper to first layer.
            cell = tf.contrib.rnn.AttentionCellWrapper(
                cell, attn_length, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell


def get_build_graph_fn(encoder_decoder,
                        sequence_example_file_paths=None,
                        batch_size = BATCH_SIZE,
                        mode = 'train'):
    """Returns a function that builds the TensorFlow graph.

    Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation.

    Returns:
    A function that builds the TF ops when called.

    Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
    """
    if mode not in ('train', 'eval', 'generate_from_eval', 'generate_from_list'):
        raise ValueError("The mode parameter must be 'train', 'eval', "
                         "or 'generate'. The mode parameter was: %s" % mode)

    input_size = encoder_decoder.input_size
    output_size = encoder_decoder.output_size
    # print('output_size',output_size)
    # print('input_size',input_size)

    def build():
        """Builds the Tensorflow graph."""

        if mode != "generate_from_list":
            inputs, duration, silence, velocity, lengths = \
                get_padded_batch_using_quene(
                sequence_example_file_paths,
                batch_size = batch_size,
                input_size = input_size,
                shuffle = True if mode == 'train' else False)
        else:
            inputs = tf.placeholder(tf.float32, [1, None, input_size])
            duration, silence, velocity, lengths= None, None, None, None

        cell = make_rnn_cell(
            RNN_LAYERS,
            dropout_keep_prob=(
                0.7 if mode == 'train' else 1),
            attn_length=40)

        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state,
            swap_memory=True)


        outputs_flat = flatten_maybe_padded_sequences(
            outputs, lengths)
        logits_flat = tf.contrib.layers.linear(outputs_flat, output_size)

        duration_out_flat = logits_flat[:,:DURATION_RANGE]
        silence_out_flat = logits_flat[:,DURATION_RANGE:DURATION_RANGE + SILENCE_RANGE]
        velocity_out_flat = logits_flat[:,-VELOCITY_RANGE:]

        if mode != 'generate_from_list':
            duration_flat = flatten_maybe_padded_sequences(duration, lengths)
            silence_flat = flatten_maybe_padded_sequences(silence, lengths)
            velocity_flat = flatten_maybe_padded_sequences(velocity, lengths)

            if mode == 'generate_from_eval':
                tf.add_to_collection('duration_flat', duration_flat)
                tf.add_to_collection('silence_flat', silence_flat)
                tf.add_to_collection('velocity_flat', velocity_flat)

            else:
                duration_softmax_cross_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=duration_flat,
                    logits=duration_out_flat)

                silence_softmax_cross_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=silence_flat,
                    logits=silence_out_flat)

                velocity_softmax_cross_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=velocity_flat,
                    logits=velocity_out_flat)

                concat_softmax_cross_entropy = tf.concat([
                    duration_softmax_cross_entropy,
                    silence_softmax_cross_entropy,
                    velocity_softmax_cross_entropy], 0)

                duration_predictions_flat = tf.argmax(duration_out_flat, axis=1)
                silence_predictions_flat = tf.argmax(silence_out_flat, axis=1)
                velocity_predictions_flat = tf.argmax(velocity_out_flat, axis=1)

                duration_correct_predictions = tf.to_float(
                    tf.equal(duration_flat, duration_predictions_flat))
                silence_correct_predictions = tf.to_float(
                    tf.equal(silence_flat, silence_predictions_flat))
                velocity_correct_predictions = tf.to_float(
                    tf.equal(velocity_flat, velocity_predictions_flat))

                num_steps =  tf.reduce_sum(lengths)
                num_steps = tf.cast(num_steps, tf.float32)

                if mode == 'train':

                    total_loss = tf.reduce_mean(concat_softmax_cross_entropy) * 3
                    total_loss_per_step = total_loss/num_steps
                    perplexity_per_step = tf.exp(total_loss_per_step)
                    perplexity = tf.exp(total_loss)

                    duration_accuracy = tf.reduce_mean(duration_correct_predictions)
                    silence_accuracy = tf.reduce_mean(silence_correct_predictions)
                    velocity_accuracy = tf.reduce_mean(velocity_correct_predictions)

                    optimizer = tf.train.AdamOptimizer(learning_rate=0.00005,
                                                        epsilon=0.000001)

                    train_op = tf.contrib.slim.learning.create_train_op(
                        total_loss, optimizer, clip_gradient_norm=3)
                    tf.add_to_collection('train_op', train_op)

                    vars_to_summarize = {
                        'loss': total_loss,
                        'metrics/loss_per_step': total_loss_per_step,
                        'metrics/perplexity_per_step': perplexity_per_step,
                        'metrics/perplexity': perplexity,
                        'metrics/duration_accuracy': duration_accuracy,
                        'metrics/silence_accuracy': silence_accuracy,
                        'metrics/velocity_accuracy': velocity_accuracy,
                    }
                elif mode == 'eval':
                    vars_to_summarize, update_ops =\
                        tf.contrib.metrics.aggregate_metric_map(
                        {
                        'loss': tf.metrics.mean(3*concat_softmax_cross_entropy),
                        'metrics/duration_accuracy': tf.metrics.accuracy(
                            duration_flat, duration_predictions_flat),
                        'metrics/silence_accuracy': tf.metrics.accuracy(
                            silence_flat, silence_predictions_flat),
                        'metrics/velocity_accuracy': tf.metrics.accuracy(
                            velocity_flat, velocity_predictions_flat),
                        'metrics/loss_per_step': tf.metrics.mean(
                            tf.reduce_sum(3*concat_softmax_cross_entropy)/num_steps,
                            weights=num_steps),
                            }
                        )
                    for updates_op in update_ops.values():
                        tf.add_to_collection('eval_ops', updates_op)

                    vars_to_summarize['metrics/perplexity'] = tf.exp(
                        vars_to_summarize['loss'])
                    vars_to_summarize['metrics/perplexity_per_step'] = tf.exp(
                        vars_to_summarize['metrics/loss_per_step'])

                for var_name, var_value in six.iteritems(vars_to_summarize):
                    tf.summary.scalar(var_name, var_value)
                    tf.add_to_collection(var_name, var_value)

        if mode.startswith('generate'):
            temperature = tf.placeholder(tf.float32, [])
            duration_softmax = tf.nn.softmax(
                tf.div(duration_out_flat, tf.fill([DURATION_RANGE], temperature)))
            silence_softmax = tf.nn.softmax(
                tf.div(silence_out_flat, tf.fill([SILENCE_RANGE], temperature)))
            velocity_softmax = tf.nn.softmax(
                tf.div(velocity_out_flat, tf.fill([VELOCITY_RANGE], temperature)))
            generate_dict = {
                'inputs': inputs,
                'initial_state': initial_state,
                'final_state': final_state,
                'temperature': temperature,
                'duration_softmax': duration_softmax,
                'silence_softmax': silence_softmax,
                'velocity_softmax': velocity_softmax,
            }

            for var_name, var_value in six.iteritems(generate_dict):
                tf.add_to_collection(var_name, var_value)

    return build


def run_training(build_graph_fn, train_dir, num_training_steps=None,
                 summary_frequency=10, save_checkpoint_secs=60,
                 checkpoints_to_keep=10, master='', task=0, num_ps_tasks=0):
    """Runs the training loop.

    Args:
    build_graph_fn: A function that builds the graph ops.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console and
        written to disk.
    save_checkpoint_secs: The frequency at which to save checkpoints, in
        seconds.
    checkpoints_to_keep: The number of most recent checkpoints to keep in
       `train_dir`. Keeps all if set to 0.
    master: URL of the Tensorflow master.
    task: Task number for this worker.
    num_ps_tasks: Number of parameter server tasks.
    """
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
            build_graph_fn()

            global_step = tf.train.get_or_create_global_step()
            loss = tf.get_collection('loss')[0]
            perplexity = tf.get_collection('metrics/perplexity')[0]
            duration_accuracy = tf.get_collection('metrics/duration_accuracy')[0]
            silence_accuracy = tf.get_collection('metrics/silence_accuracy')[0]
            velocity_accuracy = tf.get_collection('metrics/velocity_accuracy')[0]
            train_op = tf.get_collection('train_op')[0]

            logging_dict = {
              'Global Step': global_step,
              'Loss': loss,
              'Perplexity': perplexity,
              'duration_accuracy': duration_accuracy,
              'silence_accuracy': silence_accuracy,
              'velocity_accuracy':velocity_accuracy
            }
            hooks = [
                tf.train.NanTensorHook(loss),
                tf.train.LoggingTensorHook(
                    logging_dict, every_n_iter=summary_frequency),
                tf.train.StepCounterHook(
                    output_dir=train_dir, every_n_steps=summary_frequency),
                LossTooBigHook(loss, 13)
            ]
            if num_training_steps:
                hooks.append(tf.train.StopAtStepHook(num_training_steps))

            scaffold = tf.train.Scaffold(
                saver=tf.train.Saver(max_to_keep=checkpoints_to_keep))

            tf.logging.info('Starting training loop...')
            tf.contrib.training.train(
                train_op=train_op,
                logdir=train_dir,
                scaffold=scaffold,
                hooks=hooks,
                save_checkpoint_secs=save_checkpoint_secs,
                save_summaries_steps=summary_frequency,
                master=master,
                is_chief=task == 0)
            tf.logging.info('Training complete.')


def main():
    encoder = MelodyEncoderDecoder()
    build_graph_fn = get_build_graph_fn(encoder_decoder = encoder,
        sequence_example_file_paths=TRAINING_DATA, batch_size = BATCH_SIZE)
    run_training(build_graph_fn, TRAINING_DIR)

if __name__ == '__main__':
    main()
