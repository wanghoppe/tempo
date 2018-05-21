from training import get_build_graph_fn, TRAINING_DIR, EVAL_DIR, TRAINING_DATA
from training import BATCH_SIZE
from encoder_decoder import MelodyEncoderDecoder
import tensorflow as tf
from magenta.common import count_records

EVAL_DATA = ['/home/hoppe/Code/tempo/training_data/eval_melodies.tfrecord']

class EvalLoggingTensorHook(tf.train.LoggingTensorHook):
  """A revised version of LoggingTensorHook to use during evaluation.

  This version supports being reset and increments `_iter_count` before run
  instead of after run.
  """

  def begin(self):
    # Reset timer.
    self._timer.update_last_triggered_step(0)
    super(EvalLoggingTensorHook, self).begin()

  def before_run(self, run_context):
    self._iter_count += 1
    return super(EvalLoggingTensorHook, self).before_run(run_context)

  def after_run(self, run_context, run_values):
    super(EvalLoggingTensorHook, self).after_run(run_context, run_values)
    self._iter_count -= 1

def run_eval(build_graph_fn, train_dir, eval_dir, num_batches,
             timeout_secs=300):
    """Runs the training loop.

    Args:
    build_graph_fn: A function that builds the graph ops.
    train_dir: The path to the directory where checkpoints will be loaded
        from for evaluation.
    eval_dir: The path to the directory where the evaluation summary events
        will be written to.
    num_batches: The number of full batches to use for each evaluation step.
    timeout_secs: The number of seconds after which to stop waiting for a new
        checkpoint.
    """
    with tf.Graph().as_default():
        build_graph_fn()

        global_step = tf.train.get_or_create_global_step()
        loss = tf.get_collection('loss')[0]
        loss_per_step = tf.get_collection('metrics/loss_per_step')[0]
        perplexity = tf.get_collection('metrics/perplexity')[0]
        duration_accuracy = tf.get_collection('metrics/duration_accuracy')[0]
        silence_accuracy = tf.get_collection('metrics/silence_accuracy')[0]
        velocity_accuracy = tf.get_collection('metrics/velocity_accuracy')[0]
        eval_ops = tf.get_collection('eval_ops')

        logging_dict = {
          'Global Step': global_step,
          'Loss': loss,
          'loss_per_step': loss_per_step,
          'Perplexity': perplexity,
          'duration_accuracy': duration_accuracy,
          'silence_accuracy': silence_accuracy,
          'velocity_accuracy':velocity_accuracy
        }
        hooks = [
            tf.train.LoggingTensorHook(logging_dict, every_n_iter=num_batches),
            tf.contrib.training.StopAfterNEvalsHook(num_batches),
            tf.contrib.training.SummaryAtEndHook(eval_dir),
        ]

        tf.contrib.training.evaluate_once(
            train_dir,
            eval_ops=eval_ops,
            hooks=hooks)

def main():
    BATCH_SIZE = 4
    encoder = MelodyEncoderDecoder()
    build_graph_fn = get_build_graph_fn(encoder_decoder = encoder,
        sequence_example_file_paths=EVAL_DATA, mode = 'eval', batch_size = BATCH_SIZE)
    num_batches = count_records(EVAL_DATA) // BATCH_SIZE
    check_point = tf.train.latest_checkpoint(TRAINING_DIR)
    run_eval(build_graph_fn, check_point, EVAL_DIR, num_batches)

if __name__ == '__main__':
    main()
