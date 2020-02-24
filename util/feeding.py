# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as contrib_audio

from util.config import Config
from util.text import text_to_char_array
from util.flags import FLAGS
from util.spectrogram_augmentations import augment_freq_time_mask, augment_dropout, augment_pitch_and_tempo, augment_speed_up, augment_sparse_warp
from util.audio import read_frames_from_file, vad_split, DEFAULT_FORMAT


def read_csvs(csv_files):
    sets = []
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1))) # pylint: disable=cell-var-from-loop
        sets.append(file)
    # Concat all sets, drop any extra columns, re-index the final result as 0..N
    return pandas.concat(sets, join='inner', ignore_index=True)


def samples_to_mfccs(samples, sample_rate, train_phase=False):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)

    # Data Augmentations
    if train_phase:
        if FLAGS.augmentation_spec_dropout_keeprate < 1:
            spectrogram = augment_dropout(spectrogram,
                                          keep_prob=FLAGS.augmentation_spec_dropout_keeprate)

        # sparse warp must before freq/time masking
        if FLAGS.augmentation_sparse_warp:
            spectrogram = augment_sparse_warp(spectrogram,
                                              time_warping_para=FLAGS.augmentation_sparse_warp_time_warping_para,
                                              interpolation_order=FLAGS.augmentation_sparse_warp_interpolation_order,
                                              regularization_weight=FLAGS.augmentation_sparse_warp_regularization_weight,
                                              num_boundary_points=FLAGS.augmentation_sparse_warp_num_boundary_points,
                                              num_control_points=FLAGS.augmentation_sparse_warp_num_control_points)

        if FLAGS.augmentation_freq_and_time_masking:
            spectrogram = augment_freq_time_mask(spectrogram,
                                                 frequency_masking_para=FLAGS.augmentation_freq_and_time_masking_freq_mask_range,
                                                 time_masking_para=FLAGS.augmentation_freq_and_time_masking_time_mask_range,
                                                 frequency_mask_num=FLAGS.augmentation_freq_and_time_masking_number_freq_masks,
                                                 time_mask_num=FLAGS.augmentation_freq_and_time_masking_number_time_masks)

        if FLAGS.augmentation_pitch_and_tempo_scaling:
            spectrogram = augment_pitch_and_tempo(spectrogram,
                                                  max_tempo=FLAGS.augmentation_pitch_and_tempo_scaling_max_tempo,
                                                  max_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_max_pitch,
                                                  min_pitch=FLAGS.augmentation_pitch_and_tempo_scaling_min_pitch)

        if FLAGS.augmentation_speed_up_std > 0:
            spectrogram = augment_speed_up(spectrogram, speed_std=FLAGS.augmentation_speed_up_std)

    mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                               sample_rate=sample_rate,
                               dct_coefficient_count=Config.n_input,
                               upper_frequency_limit=FLAGS.audio_sample_rate/2)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features(wav_filename, train_phase=False):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate, train_phase=train_phase)

    if train_phase:
        if FLAGS.data_aug_features_multiplicative > 0:
            features = features*tf.random.normal(mean=1, stddev=FLAGS.data_aug_features_multiplicative, shape=tf.shape(features))

        if FLAGS.data_aug_features_additive > 0:
            features = features+tf.random.normal(mean=0.0, stddev=FLAGS.data_aug_features_additive, shape=tf.shape(features))

    return features, features_len


def entry_to_features(wav_filename, transcript, train_phase):
    # https://bugs.python.org/issue32117
    features, features_len = audiofile_to_features(wav_filename, train_phase=train_phase)
    return wav_filename, features, features_len, tf.SparseTensor(*transcript)


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset_sorted(csvs):
    '''
    Creates a sorted verstion of data set which should be passed to 
    create_dataset() function if sorting is desired.
    '''

    df = read_csvs(csvs)
    df.sort_values(by='wav_filesize', inplace=True)

    df['transcript'] = df.apply(text_to_char_array, alphabet=Config.alphabet, result_type='reduce', axis=1)

    return df

########################CURRICULUM LEARNING PART ##############################

def curriculum_sampling(dataframe, batch_size, split_array=[0.5, 0.5]):    
    '''
    Creates a new dataframe according to a distribution
    Params: 
        dataframe: csv_dataframe loaded into memory
        batch_size: (int) num samples in a batch
        split_array: list the proportion of taking samples according
                    to lm_ranking and compression ranking e.g. [0.3, 0.7]
    Returns: dataframe with adjusted probabilities.
    '''

    df = read_csvs(csvs)
    df['transcript'] = df.apply(text_to_char_array, alphabet=Config.alphabet, result_type='reduce', axis=1)

    df = dataframe.rename(columns={"path": "wav_filename", 
                            "sentence": "transcription"})
    
    cols_to_keep = ['wav_filename', 'transcription', 
                    'lm_scores', 
                    'compression_scores',
                    'harmonic_mean_scores']
    
    df = df[cols_to_keep]
    
    #50/50 split
    batches = pd.DataFrame(columns=['wav_filename', 
                                    'lm_scores', 
                                    'compression_scores',
                                    'harmonic_mean_scores', 
                                    'transcription'
                                   ])
    
    while len(df) > batch_size:
        lm_chunk = df.sample(n=int(batch_size*split_array[0]), 
                         replace=False, weights='lm_scores', axis=0)
        df = df.drop(lm_chunk.index)
        comp_chunk = df.sample(n=int(batch_size*split_array[1]), 
                         replace=False, weights='compression_scores', axis=0)
        df = df.drop(comp_chunk.index)
        
        batch = pd.concat([lm_chunk, comp_chunk], axis=0, ignore_index=True)

        batches = pd.concat([batches, batch], axis=0, ignore_index=True)
        
    if len(df) > 0:
        batches = pd.concat([batches, df], axis=0, ignore_index=True)
        
    return batches

def alpha(z, f):
    '''
    Calculates alpha score to be subtracted from original
    lm, hm, compression scores.
    '''
    def func(z):
        z = np.asarray(z)
        return -np.sqrt((1-z)*(1+z)) + 1
    z = np.asarray(z)
    offset = func([0])
    return f*func(z) + offset

def update_scores(dataframe, loss_old, loss_new):
    '''
    Updates lm, hm, compression scores per epoch 
    starting the second epoch
    Params:
        dataframe: (pd.DataFrame) resampled curriculum dataframe
        loss_old: (float) per epoch loss from the previous epoch
        loss_new: (float) per epoch loss from the next epoch
    '''

    f_factor = min(1-((loss_old-loss_new)/loss_old),0.99)    
    #Update lm scores
    #alphas_lm = dataframe['lm_scores']*0.1
    alphas_lm = alpha(dataframe['lm_scores'], f_factor)
    updated_lm_scores = dataframe['lm_scores'] - alphas_lm
    dataframe['lm_scores'] = updated_lm_scores
    #Normalize lm scores
    lm_sum = sum(updated_lm_scores)
    print(lm_sum)
    dataframe['lm_scores'] = [score/lm_sum for score in updated_lm_scores]
    
    #Update compression scores
    #alphas_comp = dataframe['compression_scores']*f_factor
    alphas_comp = alpha(dataframe['compression_scores'], f_factor)
    updated_comp_scores = dataframe['compression_scores'] - alphas_comp
    dataframe['compression_scores'] = updated_comp_scores
    comp_sum = sum(updated_comp_scores)
    dataframe['compression_scores'] = [score/comp_sum for score in updated_comp_scores]
    
    #print(alphas_lm, alphas_comp)
    #update HM
    alphas_hm = dataframe['harmonic_mean_scores']*f_factor
    updated_hm_scores = dataframe['harmonic_mean_scores'] - alphas_hm
    dataframe['harmonic_mean_scores'] = updated_comp_scores
    hm_sum = sum(updated_hm_scores)
    dataframe['harmonic_mean_scores'] = [score/hm_sum for score in updated_hm_scores]
    
    
    return dataframe

###########################################################################################

#Changed reading
def create_dataset(dataframe, batch_size, enable_cache=False, cache_path=None, train_phase=False):
    ## Loading csv files to dcreate_dataset    
    #df = read_csvs(csvs)
    df['transcript'] = df.apply(text_to_char_array, alphabet=Config.alphabet, result_type='reduce', axis=1)

    def generate_values():
        for _, row in df.iterrows():
            yield row.wav_filename, to_sparse_tuple(row.transcript)

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(wav_filenames, features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size,
                                         padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        wav_filenames = wav_filenames.batch(batch_size)
        return tf.data.Dataset.zip((wav_filenames, features, transcripts))

    num_gpus = len(Config.available_devices)
    process_fn = partial(entry_to_features, train_phase=train_phase)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                              .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE))

    if enable_cache:
        dataset = dataset.cache(cache_path)

    dataset = (dataset.window(batch_size, drop_remainder=True).flat_map(batch_fn)
                      .prefetch(num_gpus))

    return dataset


def split_audio_file(audio_path,
                     audio_format=DEFAULT_FORMAT,
                     batch_size=1,
                     aggressiveness=3,
                     outlier_duration_ms=10000,
                     outlier_batch_size=1):
    sample_rate, _, sample_width = audio_format
    multiplier = 1.0 / (1 << (8 * sample_width - 1))

    def generate_values():
        frames = read_frames_from_file(audio_path)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            segment_buffer, time_start, time_end = segment
            samples = np.frombuffer(segment_buffer, dtype=np.int16)
            samples = samples * multiplier
            samples = np.expand_dims(samples, axis=1)
            yield time_start, time_end, samples

    def to_mfccs(time_start, time_end, samples):
        features, features_len = samples_to_mfccs(samples, sample_rate)
        return time_start, time_end, features, features_len

    def create_batch_set(bs, criteria):
        return (tf.data.Dataset
                .from_generator(generate_values, output_types=(tf.int32, tf.int32, tf.float32))
                .map(to_mfccs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .filter(criteria)
                .padded_batch(bs, padded_shapes=([], [], [None, Config.n_input], [])))

    nds = create_batch_set(batch_size,
                           lambda start, end, f, fl: end - start <= int(outlier_duration_ms))
    ods = create_batch_set(outlier_batch_size,
                           lambda start, end, f, fl: end - start > int(outlier_duration_ms))
    dataset = nds.concatenate(ods)
    dataset = dataset.prefetch(len(Config.available_devices))
    return dataset


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
