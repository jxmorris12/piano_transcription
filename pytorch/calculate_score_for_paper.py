import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../../autoth'))
from autoth.core import HyperParamsOptimizer
import numpy as np
import argparse
import librosa
import mir_eval
import torch
import time
import h5py
import pickle 
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor
 
from utilities import (create_folder, get_filename, traverse_folder, 
    int16_to_float32, TargetProcessor, RegressionPostProcessor, note_to_freq)
import config
from inference import PianoTranscription


def infer_prob(args):
    """Inference the output probabilites of MAESTRO dataset.

    Args:
      cuda: bool
      audio_path: str
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    probs_dir = args.probs_dir
    split = args.split
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note
    dataset = 'maestro'

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    create_folder(probs_dir)

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples)

    (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

    n = 0
    for n, hdf5_path in enumerate(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            if hf.attrs['split'].decode() == split:
                print(n, hdf5_path)
                n += 1

                # Load audio                
                audio = int16_to_float32(hf['waveform'][:])
                midi_events = [e.decode() for e in hf['midi_event'][:]]
                midi_events_time = hf['midi_event_time'][:]
        
                # Ground truths processor
                target_processor = TargetProcessor(
                    segment_seconds=len(audio) / sample_rate, 
                    frames_per_second=frames_per_second, begin_note=begin_note, 
                    classes_num=classes_num)

                # Get ground truths
                (target_dict, note_events, pedal_events) = \
                    target_processor.process(start_time=0, 
                        midi_events_time=midi_events_time, 
                        midi_events=midi_events, extend_pedal=True)

                ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                ref_velocity = np.array([event['velocity'] for event in note_events])

                # Transcribe
                transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
                output_dict = transcribed_dict['output_dict']

                # Pack probabilites to dump
                total_dict = {key: output_dict[key] for key in output_dict.keys()}
                total_dict['frame_roll'] = target_dict['frame_roll']
                total_dict['ref_on_off_pairs'] = ref_on_off_pairs
                total_dict['ref_midi_notes'] = ref_midi_notes
                total_dict['ref_velocity'] = ref_velocity

                if 'pedal_frame_output' in output_dict.keys():
                    total_dict['ref_pedal_on_off_pairs'] = \
                        np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
                    total_dict['pedal_frame_roll'] = target_dict['pedal_frame_roll']
                    
                prob_path = os.path.join(probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
                create_folder(os.path.dirname(prob_path))
                pickle.dump(total_dict, open(prob_path, 'wb'))


class RegressionScoreCalculator(object):
    def __init__(self, hdf5s_dir, probs_dir, split):
        """Evaluate piano transcription metrics of the post processed 
        pre-calculated system outputs.
        """
        self.split = split
        self.probs_dir = probs_dir
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.velocity_scale = config.velocity_scale
        self.velocity = False
        self.pedal = True

        self.evaluate_frame = True
        self.onset_tolerance = 0.05
        self.offset_ratio = None
        self.offset_min_tolerance = 0.05
        self.pedal_offset_threshold = 0.2
        
        (hdf5_names, self.hdf5_paths) = traverse_folder(hdf5s_dir)

    def __call__(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """
        stats_dict = self.metrics(params)
        return np.mean(stats_dict['f1'])

    def metrics(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """
        n = 0
        list_args = []

        for n, hdf5_path in enumerate(self.hdf5_paths):
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == self.split:
                    list_args.append([n, hdf5_path, params])
                    """e.g., [0, 'xx.h5', [0.3, 0.3, 0.3]]"""
           
        debug = False
        if debug:
            list_args = list_args[0 :] 
            for i in range(len(list_args)):
                print(i)
                self.calculate_score_per_song(list_args[i])
            import crash
            asdf

        # Calculate metrics in parallel
        with ProcessPoolExecutor() as exector:
            results = exector.map(self.calculate_score_per_song, list_args)

        stats_list = list(results)
        stats_dict = {}
        for key in stats_list[0].keys():
            stats_dict[key] = [e[key] for e in stats_list if key in e.keys()]
        
        return stats_dict

    def calculate_score_per_song(self, args):
        """Calculate score per song.

        Args:
          args: [n, hdf5_path, params]
        """
        n = args[0]
        hdf5_path = args[1]
        [onset_threshold, offset_threshold, frame_threshold] = args[2]

        return_dict = {}

        # Load pre-calculated system outputs and ground truths
        prob_path = os.path.join(self.probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
        total_dict = pickle.load(open(prob_path, 'rb'))

        ref_on_off_pairs = total_dict['ref_on_off_pairs']
        ref_midi_notes = total_dict['ref_midi_notes']
        output_dict = total_dict

        # Calculate frame metric
        if self.evaluate_frame:
            frame_threshold = frame_threshold
            y_pred = (np.sign(total_dict['frame_output'] - frame_threshold) + 1) / 2
            y_pred[np.where(y_pred==0.5)] = 0
            y_true = total_dict['frame_roll']
            y_pred = y_pred[0 : y_true.shape[0], :]
            y_true = y_true[0 : y_pred.shape[0], :]

            tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
            return_dict['frame_precision'] = tmp[0][1]
            return_dict['frame_recall'] = tmp[1][1]
            return_dict['frame_f1'] = tmp[2][1]

        # Post processor
        post_processor = RegressionPostProcessor(self.frames_per_second, 
            classes_num=self.classes_num, onset_threshold=onset_threshold, 
            offset_threshold=offset_threshold, 
            frame_threshold=frame_threshold, 
            pedal_offset_threshold=self.pedal_offset_threshold)

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            post_processor.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'], 
                threshold=onset_threshold, neighbour=2)
        output_dict['onset_output'] = onset_output
        output_dict['onset_shift_output'] = onset_shift_output

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            post_processor.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'], 
                threshold=offset_threshold, neighbour=4)
        output_dict['offset_output'] = offset_output
        output_dict['offset_shift_output'] = offset_shift_output

        # Detect piano notes from output_dict
        est_on_off_note_vels = post_processor.output_dict_to_detected_notes(output_dict)
        est_on_offs = est_on_off_note_vels[:, 0 : 2]
        est_midi_notes = est_on_off_note_vels[:, 2]
        est_vels = est_on_off_note_vels[:, 3] * self.velocity_scale

        # Calculate note metrics
        if self.velocity:
            (note_precision, note_recall, note_f1, _) = (
                   mir_eval.transcription_velocity.precision_recall_f1_overlap(
                       ref_intervals=ref_on_off_pairs,
                       ref_pitches=note_to_freq(ref_midi_notes),
                       ref_velocities=total_dict['ref_velocity'],
                       est_intervals=est_on_offs,
                       est_pitches=note_to_freq(est_midi_notes),
                       est_velocities=est_vels,
                       onset_tolerance=self.onset_tolerance, 
                       offset_ratio=self.offset_ratio, 
                       offset_min_tolerance=self.offset_min_tolerance))
        else:
            note_precision, note_recall, note_f1, _ = \
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs, 
                    ref_pitches=note_to_freq(ref_midi_notes), 
                    est_intervals=est_on_offs, 
                    est_pitches=note_to_freq(est_midi_notes), 
                    onset_tolerance=self.onset_tolerance, 
                    offset_ratio=self.offset_ratio, 
                    offset_min_tolerance=self.offset_min_tolerance)

        if self.pedal:
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = \
                post_processor.get_binarized_output_from_regression(
                    reg_output=output_dict['reg_pedal_offset_output'], 
                    threshold=self.pedal_offset_threshold, neighbour=4)
            output_dict['pedal_offset_output'] = pedal_offset_output

            # Detect piano notes from output_dict
            ref_pedal_on_off_pairs = output_dict['ref_pedal_on_off_pairs']
            est_pedal_on_offs = post_processor.output_dict_to_detected_pedals(output_dict)

            # Calculate pedal metrics
            if len(ref_pedal_on_off_pairs) > 0:
                pedal_precision, pedal_recall, pedal_f1, _ = \
                    mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_pedal_on_off_pairs, 
                        ref_pitches=np.ones(ref_pedal_on_off_pairs.shape[0]), 
                        est_intervals=est_pedal_on_offs, 
                        est_pitches=np.ones(est_pedal_on_offs.shape[0]), 
                        onset_tolerance=0.2)

                return_dict['pedal_f1'] = pedal_f1

                y_pred = (np.sign(total_dict['pedal_frame_output'] - 0.5) + 1) / 2
                y_pred[np.where(y_pred==0.5)] = 0
                y_true = total_dict['pedal_frame_roll']
                y_pred = y_pred[0 : y_true.shape[0]]
                y_true = y_true[0 : y_pred.shape[0]]
                
                tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
                return_dict['pedal_frame_precision'] = tmp[0][1]
                return_dict['pedal_frame_recall'] = tmp[1][1]
                return_dict['pedal_frame_f1'] = tmp[2][1]

                print('pedal f1: {:.3f}, frame f1: {:.3f}'.format(pedal_f1, return_dict['pedal_frame_f1']))

        print('note f1: {:.3f}'.format(note_f1))

        return_dict['note_precision'] = note_precision
        return_dict['note_recall'] = note_recall
        return_dict['note_f1'] = note_f1
        return return_dict


def calculate_metrics(args, thresholds=None):

    # Arugments & parameters
    workspace = args.workspace
    probs_dir = args.probs_dir
    split = args.split
    dataset = 'maestro'
    
    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    # Score calculator
    score_calculator = RegressionScoreCalculator(hdf5s_dir, probs_dir, split=split)

    if not thresholds:
        thresholds = [0.3, 0.3, 0.3]
    else:
        pass

    t1 = time.time()
    stats_dict = score_calculator.metrics(thresholds)
    print('Time: {:.3f}'.format(time.time() - t1))
    
    for key in stats_dict.keys():
        print('{}: {:.4f}'.format(key, np.mean(stats_dict[key])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_infer_prob = subparsers.add_parser('infer_prob')
    parser_infer_prob.add_argument('--workspace', type=str, required=True)
    parser_infer_prob.add_argument('--model_type', type=str, required=True)
    parser_infer_prob.add_argument('--checkpoint_path', type=str, required=True)
    parser_infer_prob.add_argument('--probs_dir', type=str, required=True)
    parser_infer_prob.add_argument('--split', type=str, required=True)
    parser_infer_prob.add_argument('--cuda', action='store_true', default=False)

    parser_metrics = subparsers.add_parser('calculate_metrics')
    parser_metrics.add_argument('--workspace', type=str, required=True)
    parser_metrics.add_argument('--probs_dir', type=str, required=True)
    parser_metrics.add_argument('--split', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'infer_prob':
        infer_prob(args)

    elif args.mode == 'calculate_metrics':
        calculate_metrics(args)

    else:
        raise Exception('Incorrct argument!')