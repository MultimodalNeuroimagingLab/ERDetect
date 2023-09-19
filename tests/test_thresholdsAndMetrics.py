"""
This test script processes all validation datasets and tries evoked response detection on each dataset using
the standard deviation from baseline detection method with a range of different thresholds (factor x std. dev).
In addition, it also returns all of the metrics (waveform and cross-projection).

The test output for each dataset is stored as a .mat file. Matlab scripts are provided to visualize the results.


=====================================================
Copyright 2023, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import os
from ieegprep.bids.sidecars import load_channel_info, load_elec_stim_events
from ieegprep.bids.data_epoch import load_data_epochs_averages
from erdetect.core.config import get as cfg, set
from erdetect.core.metrics.metric_cross_proj import MetricCrossProj
from erdetect.core.metrics.metric_waveform import MetricWaveform
from erdetect.core.detection import ieeg_detect_er
import numpy as np
import scipy.io as sio



output_path = 'D:\\BIDS_erdetect\\derivatives\\app_thresholdsAndMetrics\\'
t_factors = range(1, 1501, 1)


subjects = dict()
subjects['UMCU20'] = "D:\BIDS_erdetect\sub-UMCU20\ses-1\ieeg\sub-UMCU20_ses-1_task-SPESclin_run-011757_ieeg.vhdr"
subjects['UMCU21'] = "D:\BIDS_erdetect\sub-UMCU21\ses-1\ieeg\sub-UMCU21_ses-1_task-SPESclin_run-021525_ieeg.vhdr"
subjects['UMCU22'] = "D:\BIDS_erdetect\sub-UMCU22\ses-1\ieeg\sub-UMCU22_ses-1_task-SPESclin_run-011714_ieeg.vhdr"
subjects['UMCU23'] = "D:\BIDS_erdetect\sub-UMCU23\ses-1\ieeg\sub-UMCU23_ses-1_task-SPESclin_run-021706_ieeg.vhdr"
subjects['UMCU25'] = "D:\BIDS_erdetect\sub-UMCU25\ses-1\ieeg\sub-UMCU25_ses-1_task-SPESclin_run-031729_ieeg.vhdr"
subjects['UMCU26'] = "D:\BIDS_erdetect\sub-UMCU26\ses-1\ieeg\sub-UMCU26_ses-1_task-SPESclin_run-011555_ieeg.vhdr"
subjects['UMCU59'] = "D:\BIDS_erdetect\sub-UMCU59\ses-1\ieeg\sub-UMCU59_ses-1_task-SPESclin_run-041501_ieeg.vhdr"
subjects['UMCU62'] = "D:\BIDS_erdetect\sub-UMCU62\ses-1b\ieeg\sub-UMCU62_ses-1b_task-SPESclin_run-050941_ieeg.vhdr"
subjects['UMCU67'] = "D:\BIDS_erdetect\sub-UMCU67\ses-1\ieeg\sub-UMCU67_ses-1_task-SPESclin_run-021704_ieeg.vhdr"
subjects['MAYO01'] = "D:\BIDS_erdetect\sub-MAYO01\ses-ieeg01\ieeg\sub-MAYO01_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
subjects['MAYO02'] = "D:\BIDS_erdetect\sub-MAYO02\ses-ieeg01\ieeg\sub-MAYO02_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
subjects['MAYO03'] = "D:\BIDS_erdetect\sub-MAYO03\ses-ieeg01\ieeg\sub-MAYO03_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
subjects['MAYO04'] = "D:\BIDS_erdetect\sub-MAYO04\ses-ieeg01\ieeg\sub-MAYO04_ses-ieeg01_task-ccep_run-01_ieeg.mefd"
subjects['MAYO05'] = "D:\BIDS_erdetect\sub-MAYO05\ses-ieeg01\ieeg\sub-MAYO05_ses-ieeg01_task-ccep_run-01_ieeg.mefd"

#
output_dict = dict()
output_dict['subjects'] = subjects
for (subject, subject_path) in subjects.items():

    bids_subset_data_path = os.path.abspath(os.path.expanduser(os.path.expandvars(subject_path)))
    bids_subset_root = bids_subset_data_path[:bids_subset_data_path.rindex('_')]

    # load channels
    try:
        channel_tsv = load_channel_info(bids_subset_root + '_channels.tsv')
    except (FileNotFoundError, LookupError):
        raise RuntimeError('Could not load the channel metadata')


    # pick channels
    channels_incl = []
    channels_measured_incl = []
    channels_stim_incl = []

    channels_have_status = 'status' in channel_tsv.columns
    for index, row in channel_tsv.iterrows():
        if channels_have_status and row['status'].lower() == 'bad':
            continue
        if row['type'].upper() in cfg('channels', 'measured_types'):
            channels_measured_incl.append(row['name'])
        if row['type'].upper() in cfg('channels', 'stim_types'):
            channels_stim_incl.append(row['name'])

    # read events
    try:
        trial_onsets, trial_pairs, stim_pairs_onsets, bad_trial_onsets = load_elec_stim_events(bids_subset_root + '_events.tsv',
                                                                                                exclude_bad_events=True,
                                                                                                concat_bidirectional_stimpairs=cfg('trials', 'concat_bidirectional_pairs'),
                                                                                                only_stimpairs_between_channels=channels_stim_incl)
    except (RuntimeError):
        raise RuntimeError('Could not load the electrical stimulation event metadata')

    # remove stim-pairs with too little trials
    stimpair_remove_keys = []
    for stim_pair, onsets in stim_pairs_onsets.items():
        if len(onsets) < cfg('trials', 'minimum_stimpair_trials'):
            stimpair_remove_keys.append(stim_pair)
    if len(stimpair_remove_keys) > 0:
        for stim_pair in stimpair_remove_keys:
            del stim_pairs_onsets[stim_pair]

    # load
    try:

        metric_callbacks = tuple()
        metric_callbacks += tuple([MetricCrossProj.process_callback])
        metric_callbacks += tuple([MetricWaveform.process_callback])

        sampling_rate, averages, metrics = load_data_epochs_averages(bids_subset_data_path, channels_measured_incl, stim_pairs_onsets,
                                                                     trial_epoch=cfg('trials', 'trial_epoch'),
                                                                     baseline_norm=cfg('trials', 'baseline_norm'),
                                                                     baseline_epoch=cfg('trials', 'baseline_epoch'),
                                                                     out_of_bound_handling=cfg('trials', 'out_of_bounds_handling'),
                                                                     metric_callbacks=metric_callbacks,
                                                                     high_pass=cfg('preprocess', 'high_pass'),
                                                                     early_reref=None,
                                                                     line_noise_removal=None,
                                                                     late_reref=None,
                                                                     preproc_priority='speed')

        cross_proj_metrics = np.array(metrics[:, :, 0].tolist())
        waveform_metrics = np.array(metrics[:, :, 1].tolist())

    except (ValueError, RuntimeError):
        raise RuntimeError('Could not load data')

    # for each stimulation pair condition, NaN out the values of the measured electrodes that were stimulated
    for stim_pair_index, stim_pair in enumerate(stim_pairs_onsets):
        stim_pair_electrode_names = stim_pair.split('-')

        try:
            averages[channels_measured_incl.index(stim_pair_electrode_names[0]), stim_pair_index, :] = np.nan
            cross_proj_metrics[channels_measured_incl.index(stim_pair_electrode_names[0]), stim_pair_index] = np.nan
            waveform_metrics[channels_measured_incl.index(stim_pair_electrode_names[0]), stim_pair_index] = np.nan
        except ValueError:
            pass

        try:
            averages[channels_measured_incl.index(stim_pair_electrode_names[1]), stim_pair_index, :] = np.nan
            cross_proj_metrics[channels_measured_incl.index(stim_pair_electrode_names[1]), stim_pair_index] = np.nan
            waveform_metrics[channels_measured_incl.index(stim_pair_electrode_names[1]), stim_pair_index] = np.nan
        except ValueError:
            pass

    # determine the sample of stimulus onset (counting from the epoch start)
    onset_sample = int(round(abs(cfg('trials', 'trial_epoch')[0] * sampling_rate)))

    # try with different factors
    subject_dict = dict()
    subject_dict['subject_id'] = subject
    subject_dict['channels_measured'] = channels_measured_incl
    subject_dict['stimpairs'] = list(stim_pairs_onsets.keys())
    subject_dict['averages'] = averages
    MetricCrossProj.append_output_dict_callback(subject_dict, cross_proj_metrics)
    MetricWaveform.append_output_dict_callback(subject_dict, waveform_metrics)
    subject_dict['onset_sample'] = onset_sample
    subject_dict['range'] = []
    for t in t_factors:
        float_t = t / 100

        set(float_t, 'detection', 'std_base', 'baseline_threshold_factor')
        try:
            neg_peak_latency, er_neg_peak_amplitudes = ieeg_detect_er(averages, onset_sample, int(sampling_rate))
        except (ValueError, RuntimeError):
            raise RuntimeError('Evoked response detection failed')

        subject_dict['range'].append(float_t)
        subject_dict['neg_lat_fact_' + str(t)] = neg_peak_latency

    # intermediate saving of the data and evoked response detection results as .mat
    sio.savemat((output_path + 'sub-' + subject + '_ROC.mat'), subject_dict)
