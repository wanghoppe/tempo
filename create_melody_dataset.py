from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2
from magenta.pipelines import statistics
from magenta.pipelines import pipelines_common

from melody_pipeline import MelodyExtractorInfo
from encoder_decoder import MelodyEncoderDecoder
from melody_lib import MelodySequence, STEPS_PER_SECOND
from encoder_decoder import LOWEST_MIDI_PITCH, HIGHEST_MIDI_PITCH
import tensorflow as tf


INPUT_DIR = '/home/hoppe/Code/tempo/dataset/out.tfrecord'
OUTPUT_DIR = 'training_data'

# Stretch by -10%, 0%, 10%, and 15%.
stretch_factors = [0.9, 1.0, 1.1, 1.15]

transposition_range = list(range(-5, 6)) + [-12, 12]

class EncoderPipeline(pipeline.Pipeline):
    """A Module that converts monophonic melodies to a model specific encoding."""

    def __init__(self, name):
        """Constructs an EncoderPipeline.

          name: A unique pipeline name.
        """
        super(EncoderPipeline, self).__init__(
                input_type=MelodySequence,
                output_type=tf.train.SequenceExample,
                name=name)
        self._melody_encoder_decoder = MelodyEncoderDecoder()

    def transform(self, melody):

        encoded = self._melody_encoder_decoder.encode(melody)
        return [encoded]


def get_pipeline(eval_ratio = 0.02):
    """Returns the Pipeline instance which creates the RNN dataset.

    Args:
    config: A MelodyRnnConfig object.
    eval_ratio: Fraction of input to set aside for evaluation set.

    Returns:
    A pipeline.Pipeline instance.
    """
    partitioner = pipelines_common.RandomPartition(
            music_pb2.NoteSequence,
            ['eval_melodies', 'training_melodies'],
            [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    for mode in ['eval', 'training']:
        stretch_pipeline = note_sequence_pipelines.StretchPipeline(
            stretch_factors, name='StretchPipeline_' + mode)

        quantizer = note_sequence_pipelines.Quantizer(
            steps_per_second=STEPS_PER_SECOND, name='Quantizer_' + mode)

        transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
            transposition_range, min_pitch=LOWEST_MIDI_PITCH,
            max_pitch=HIGHEST_MIDI_PITCH, name= 'TranspositionPipeline_' + mode)

        melody_extractor = MelodyExtractorInfo(name='MelodyExtractor_' + mode)

        encoder_pipeline = EncoderPipeline(name='EncoderPipeline' + mode)

        dag[stretch_pipeline] = partitioner[mode + '_melodies']
        dag[quantizer] = stretch_pipeline
        dag[transposition_pipeline] = quantizer
        dag[melody_extractor] = transposition_pipeline
        dag[encoder_pipeline] = melody_extractor
        dag[dag_pipeline.DagOutput(mode + '_melodies')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)


def main():
    # tf.logging.set_verbosity(2)
    pipeline_instance = get_pipeline()

    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    pipeline.run_pipeline_serial(
        pipeline_instance,
        pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type),
        output_dir)

if __name__ == '__main__':
    main()
