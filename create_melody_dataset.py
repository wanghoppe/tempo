from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2
from magenta.pipelines import statistics
from magenta.pipelines import pipelines_common

from melody_pipeline import MelodyExtractor
from encoder_decoder import MelodyEncoderDecoder
from melody_lib import MelodySequence
import tensorflow as tf


INPUT_DIR = 'dataset/note_sequence.tfrecord'
OUTPUT_DIR = 'sequence_example/dataset'


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


def get_pipeline(eval_ratio = 0.1):
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
        time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
            name='TimeChangeSplitter_' + mode)
        quantizer = note_sequence_pipelines.Quantizer(
            steps_per_quarter=8, name='Quantizer_' + mode)
        melody_extractor = MelodyExtractor(name='MelodyExtractor_' + mode)
        encoder_pipeline = EncoderPipeline(name='EncoderPipeline_' + mode)

        dag[time_change_splitter] = partitioner[mode + '_melodies']
        dag[quantizer] = time_change_splitter
        dag[melody_extractor] = quantizer
        dag[encoder_pipeline] = melody_extractor
        dag[dag_pipeline.DagOutput(mode + '_melodies')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)


def main():

    pipeline_instance = get_pipeline()

    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    pipeline.run_pipeline_serial(
        pipeline_instance,
        pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type),
        output_dir)

if __name__ == '__main__':
    main()
