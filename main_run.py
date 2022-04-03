#!/usr/bin/env python3
"""
Early response detection - docker entry-point
=====================================================
Entry-point python script for the automatic detection of evoked responses in CCEP data.


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import argparse
import os
import logging
from math import isnan, ceil
from glob import glob

from bids_validator import BIDSValidator
import numpy as np
import scipy.io as sio

from app.config import load_config, write_config, get as cfg, get_config_dict, set as cfg_set, rem as cfg_rem,\
    OUTPUT_IMAGE_SIZE, LOGGING_CAPTION_INDENT_LENGTH, CONFIG_DETECTION_STD_BASE_BASELINE_EPOCH_DEFAULT, \
    CONFIG_DETECTION_STD_BASE_BASELINE_THRESHOLD_FACTOR, CONFIG_DETECTION_CROSS_PROJ_THRESHOLD, CONFIG_DETECTION_WAVEFORM_PROJ_THRESHOLD
from app.detection import ieeg_detect_er
from app.views import calc_sizes_and_fonts, calc_matrix_image_size, gen_amplitude_matrix, gen_latency_matrix
from utils.bids import load_channel_info, load_event_info, load_data_epochs_averages, RerefStruct
from utils.IeegDataReader import VALID_FORMAT_EXTENSIONS
from utils.misc import print_progressbar, is_number, CustomLoggingFormatter, multi_line_list, create_figure
from metric_callbacks import metric_cross_proj, metric_waveform



#
# version and logging
#

#
__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'version')).read()

#
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setFormatter(CustomLoggingFormatter())
logger.addHandler(logger_ch)


def log_indented_line(caption, text):
    logging.info(caption.ljust(LOGGING_CAPTION_INDENT_LENGTH, ' ') + text)


#
# define and parse the input arguments
#
parser = argparse.ArgumentParser(description='BIDS App for the automatic detection of evoked responses in CCEP data.',
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 add_help=False)
parser._positionals.title = 'Required (positional) arguments'
parser._optionals.title = 'Optional arguments'
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show this help message and exit\n\n')
parser.add_argument('bids_dir',
                    help='The directory with the input dataset formatted according to the BIDS standard.\n\n')
parser.add_argument('output_dir',
                    help='The directory where the output files should be stored. If you are running group\n'
                         'level analysis this folder should be prepopulated with the results of the\n'
                         'participant level analysis.\n\n')
parser.add_argument('--participant_label',
                    help='The label(s) of the participant(s) that should be analyzed. The label corresponds\n'
                         'to sub-<participant_label> from the BIDS spec (so it does not include \'sub-\').\n'
                         'If this parameter is not provided all subjects will be analyzed. Multiple\n'
                         'participants can be specified with a space separated list.\n\n',
                    nargs="+")
parser.add_argument('--subset_search_pattern',
                    help='The subset(s) of data that should be analyzed. The pattern should be part of a BIDS\n'
                         'compliant folder name (e.g. \'task-ccep_run-01\'). If this parameter is not provided\n'
                         'all the found subset(s) will be analyzed. Multiple subsets can be specified with\n'
                         'a space separated list.\n\n',
                    nargs="+")
parser.add_argument('--format_extension',
                    help='The data format(s) to include. The format(s) should be specified by their\n'
                         'extension (e.g. \'.edf\'). If this parameter is not provided, then by default\n'
                         'the European Data Format (\'.edf\'), BrainVision (\'.vhdr\', \'.vmrk\', \'.eeg\')\n'
                         'and MEF3 (\'.mefd\') formats will be included. Multiple formats can be specified\n'
                         'with a space separated list.\n\n',
                    nargs="+")
parser.add_argument('--config_filepath',
                    help='Configures the app according to the settings in the JSON configuration file\n\n')
parser.add_argument('--skip_bids_validator',
                    help='Skip the BIDS data-set validation\n\n',
                    action='store_true')
parser.add_argument('--preproc_prioritize_speed',
                    help='Prioritize preprocessing for speed rather than for memory. By default, while preprocessing,\n'
                         'priority is given to use as little memory as possible, which can require channel-data to be\n'
                         'retrieved twice, taking longer. This flag allows the preprocessing to keep all channel-data\n'
                         'in memory, requiring much more memory at its peak, but speeding up the process.\n'
                         'Note: In particular the speed processing when re-referencing MEF3 data will be influenced by\n'
                         '      this setting since MEF3 has the ability to read partial data from the disk, allowing for\n'
                         '      minimal memory usage. In contrast, EDF and BrainVision are loaded by MNE which holds the\n'
                         '      entire dataset in memory, so retrieval is already fast. As a result, with EDF and BrainVision,\n'
                         '      it might be counterproductive to set priority to speed since there is little gain and the\n'
                         '      memory use would double.\n\n',
                    action='store_true')
parser.add_argument('--high_pass',
                    help='Perform high-pass filtering (with a cut-off at 0.50Hz) before detection and visualization.\n'
                         'Note: If a configuration file is provided, then this command-line argument will overrule the\n'
                         '      high-pass setting in the configuration file\n\n',
                    action='store_true')
parser.add_argument('--early_reref',
                    help='Perform early re-referencing before detection and visualization. The options are:\n'
                         '      - CAR   = Common Average Re-refrencing (e.g. \'--early_reref CAR\')\n'
                         'Note: If a configuration file is provided, then this command-line argument will overrule the\n'
                         '      early re-referencing setting in the configuration file\n\n',
                    nargs="?", const='CAR')
parser.add_argument('--line_noise_removal',
                    help='Perform line-noise removal before detection and visualization. Can be either:\n'
                         '      - \'tsv\' to lookup the line-noise frequency in the BIDS channels.tsv file\n'
                         '        (e.g. \'--line_noise_removal tsv\')\n'
                         '      - set to a specific line-noise frequency (e.g. \'--line_noise_removal 60\')\n'
                         'Note: If a configuration file is provided, then this command-line argument will overrule the\n'
                         '      line-noise removal setting in the configuration file\n\n',
                    nargs="?")
parser.add_argument('--detect_positive_responses',
                    help='Detect and visualize positive evoked responses in addition to the negative responses\n\n',
                    action='store_true')
parser.add_argument('--method',
                    help='The method that should be used to detect evoked responses. the options are:\n'
                         '      - std_base   = The standard deviation of a baseline-epoch is used as a threshold\n'
                         '                     (multiplied by a factor) to determine whether the average evoked deflection\n'
                         '                     is strong enough. (e.g. \'--method std_base\')\n'
                         '      - cross-proj = Cross-projection of the trials is used to determine the inter-trial\n'
                         '                     similarity. A peak with a strong inter-trial similarity is\n'
                         '                     considered an evoked response. (e.g. \'--method cross-proj\')\n'
                         '      - waveform   = Searches for the typical (20Hz oscillation) shape of the average response\n'
                         '                     to determine whether the peak that was found can be considered an evoked.\n'
                         '                     response (e.g. \'--method waveform\')\n'
                         'Note: If a configuration file is provided, then this command-line argument will overrule the\n'
                         '      method setting in the configuration file\n\n',
                    nargs="?")
parser.add_argument('-v', '--version',
                    action='version',
                    version='ER-Detect BIDS-App version {}'.format(__version__))
args = parser.parse_args()

#
# display application information
#
log_indented_line('BIDS app:', ('Evoked Response Detection - ' + __version__))
log_indented_line('BIDS input path:', args.bids_dir)
log_indented_line('Output path:', args.output_dir)
if args.config_filepath:
    log_indented_line('Configuration file:', args.config_filepath)
logging.info('')


#
# configure
#

#  read the configuration file (if passed)
if args.config_filepath:
    if not load_config(args.config_filepath):
        logging.error('Could not load the configuration file, exiting...')
        exit(1)

# check preprocessing arguments
preproc_prioritize_speed = False
if args.preproc_prioritize_speed:
    preproc_prioritize_speed = True

if args.high_pass:
    cfg_set(True, 'preprocess', 'high_pass')

if args.line_noise_removal:
    if str(args.line_noise_removal).lower() == 'tsv':
        cfg_set('tsv', 'preprocess', 'line_noise_removal')
    elif is_number(args.line_noise_removal):
        # TODO: valid number
        cfg_set(str(args.line_noise_removal), 'preprocess', 'line_noise_removal')
    else:
        logging.error('Invalid line_noise_removal argument \'' + args.line_noise_removal + '\', either set to \'tsv\', or provide the line-noise frequency as a number.')
        exit(1)

if args.early_reref:
    if str(args.early_reref).lower() == 'car':
        cfg_set(True, 'preprocess', 'early_re_referencing', 'enabled')
        cfg_set('CAR', 'preprocess', 'early_re_referencing', 'method')
    else:
        logging.error('Invalid early_reref argument \'' + args.early_reref + '\'')
        exit(1)

# check for methodological arguments
if args.detect_positive_responses:
    cfg_set(True, 'detection', 'positive')
    cfg_set(True, 'visualization', 'positive')

# if negative or positive response visualization is enabled, make sure detection is as well
if cfg('visualization', 'negative') and not cfg('detection', 'negative'):
    logging.warning('Visualization of negative evoked responses is enabled, but the detection is set to disabled, detection of negative responses will be enabled')
    cfg_set(True, 'detection', 'negative')
if cfg('visualization', 'positive') and not cfg('detection', 'positive'):
    logging.warning('Visualization of positive evoked responses is enabled, but the detection is set to disabled, detection of positive responses will be enabled')
    cfg_set(True, 'detection', 'positive')

if args.method:
    cfg_rem('detection', 'std_base')
    cfg_rem('detection', 'cross_proj')
    cfg_rem('detection', 'waveform')
    if args.method == "std_base":
        cfg_set('std_base', 'detection', 'method')
        cfg_set(CONFIG_DETECTION_STD_BASE_BASELINE_EPOCH_DEFAULT, 'detection', 'std_base', 'baseline_epoch')
        cfg_set(CONFIG_DETECTION_STD_BASE_BASELINE_THRESHOLD_FACTOR, 'detection', 'std_base', 'baseline_threshold_factor')
    elif args.method == "cross_proj":
        cfg_set('cross_proj', 'detection', 'method')
        cfg_set(CONFIG_DETECTION_CROSS_PROJ_THRESHOLD, 'detection', 'cross_proj', 'threshold')
    elif args.method == "waveform":
        cfg_set('waveform', 'detection', 'method')
        cfg_set(CONFIG_DETECTION_WAVEFORM_PROJ_THRESHOLD, 'detection', 'waveform', 'threshold')
    else:
        logging.error('Invalid method argument \'' + args.method + '\', pick one of the following: \'std_base\', \'cross_proj\' or \'waveform\'')
        exit(1)

# if a metric is used for detection, enable them
if cfg('detection', 'method') == 'cross_proj' and not cfg('metrics', 'cross_proj', 'enabled'):
    logging.warning('Evoked response detection is set to use cross-projections but the cross-projection metric is disabled, the cross-projection metric will be enabled')
    cfg_set(True, 'metrics', 'cross_proj', 'enabled')
if cfg('detection', 'method') == 'waveform' and not cfg('metrics', 'waveform', 'enabled'):
    logging.warning('Evoked response detection is set to use waveforms but the waveform metric is disabled, the waveform metric will be enabled')
    cfg_set(True, 'metrics', 'waveform', 'enabled')

# print configuration information
log_indented_line('Preprocessing priority:', ('Speed' if preproc_prioritize_speed else 'Memory'))
log_indented_line('High-pass filtering:', ('Yes' if cfg('preprocess', 'high_pass') else 'No'))
log_indented_line('Early re-referencing:', ('Yes' if cfg('preprocess', 'early_re_referencing', 'enabled') else 'No'))
if cfg('preprocess', 'early_re_referencing', 'enabled'):
    log_indented_line('    Method:', str(cfg('preprocess', 'early_re_referencing', 'method')))
    log_indented_line('    Stim exclude epoch:', str(cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[0]) + 's : ' + str(cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[1]) + 's')
log_indented_line('Line-noise removal:', cfg('preprocess', 'line_noise_removal') + (' Hz' if is_number(cfg('preprocess', 'line_noise_removal')) else ''))
logging.info('')
log_indented_line('Trial epoch window:', str(cfg('trials', 'trial_epoch')[0]) + 's < stim onset < ' + str(cfg('trials', 'trial_epoch')[1]) + 's  (window size ' + str(abs(cfg('trials', 'trial_epoch')[1] - cfg('trials', 'trial_epoch')[0])) + 's)')
log_indented_line('Trial out-of-bounds handling:', str(cfg('trials', 'out_of_bounds_handling')))
log_indented_line('Trial baseline window:', str(cfg('trials', 'baseline_epoch')[0]) + 's : ' + str(cfg('trials', 'baseline_epoch')[1]) + 's')
log_indented_line('Trial baseline normalization:', str(cfg('trials', 'baseline_norm')))
log_indented_line('Concatenate bidirectional stimulated pairs:', ('Yes' if cfg('trials', 'concat_bidirectional_pairs') else 'No'))
log_indented_line('Minimum # of required stimulus-pair trials:', str(cfg('trials', 'minimum_stimpair_trials')))
logging.info(multi_line_list(cfg('channels', 'types'), LOGGING_CAPTION_INDENT_LENGTH, 'Channels types:', 20, ' '))
logging.info('')
log_indented_line('Cross-projection metric:', ('Enabled' if cfg('metrics', 'cross_proj', 'enabled') else 'Disabled'))
if cfg('metrics', 'cross_proj', 'enabled'):
    log_indented_line('    Cross-projection epoch:', str(cfg('metrics', 'cross_proj', 'epoch')[0]) + 's : ' + str(cfg('metrics', 'cross_proj', 'epoch')[1]) + 's')
log_indented_line('Waveform metric:', ('Enabled' if cfg('metrics', 'waveform', 'enabled') else 'Disabled'))
if cfg('metrics', 'waveform', 'enabled'):
    log_indented_line('    Waveform epoch:', str(cfg('metrics', 'waveform', 'epoch')[0]) + 's : ' + str(cfg('metrics', 'waveform', 'epoch')[1]) + 's')
    log_indented_line('    Waveform bandpass:', str(cfg('metrics', 'waveform', 'bandpass')[0]) + 'Hz - ' + str(cfg('metrics', 'waveform', 'bandpass')[1]) + 'Hz')
logging.info('')
logging.info('Detection')
log_indented_line('    Negative responses:', ('Yes' if cfg('detection', 'negative') else 'No'))
log_indented_line('    Positive responses:', ('Yes' if cfg('detection', 'positive') else 'No'))
log_indented_line('    Peak search window:', str(cfg('detection', 'peak_search_epoch')[0]) + 's : ' + str(cfg('detection', 'peak_search_epoch')[1]) + 's')
log_indented_line('    Evoked response search window:', str(cfg('detection', 'response_search_epoch')[0]) + 's : ' + str(cfg('detection', 'response_search_epoch')[1]) + 's')
log_indented_line('    Evoked response detection method:', str(cfg('detection', 'method')))
if cfg('detection', 'method') == 'std_base':
    log_indented_line('        Std baseline window:', str(cfg('detection', 'std_base', 'baseline_epoch')[0]) + 's : ' + str(cfg('detection', 'std_base', 'baseline_epoch')[1]) + 's')
    log_indented_line('        Std baseline threshold factor:', str(cfg('detection', 'std_base', 'baseline_threshold_factor')))
elif cfg('detection', 'method') == 'cross_proj':
    log_indented_line('        Cross-projection detection threshold:', str(cfg('detection', 'cross_proj', 'threshold')))
elif cfg('detection', 'method') == 'waveform':
    log_indented_line('        Waveform detection threshold:', str(cfg('detection', 'waveform', 'threshold')))
logging.info('')
logging.info('Visualization')
log_indented_line('    Negative responses:', ('Yes' if cfg('visualization', 'negative') else 'No'))
log_indented_line('    Positive responses:', ('Yes' if cfg('visualization', 'positive') else 'No'))
log_indented_line('    X-axis epoch:', str(cfg('visualization', 'x_axis_epoch')[0]) + 's : ' + str(cfg('visualization', 'x_axis_epoch')[1]) + 's')
log_indented_line('    Blank stimulation epoch:', str(cfg('visualization', 'blank_stim_epoch')[0]) + 's : ' + str(cfg('visualization', 'blank_stim_epoch')[1]) + 's')
log_indented_line('    Generate electrode images:', ('Yes' if cfg('visualization', 'generate_electrode_images') else 'No'))
log_indented_line('    Generate stimulation-pair images:', ('Yes' if cfg('visualization', 'generate_stimpair_images') else 'No'))
log_indented_line('    Generate matrix images:', ('Yes' if cfg('visualization', 'generate_matrix_images') else 'No'))
logging.info('')


#
# check if the input is a valid BIDS dataset
#

if not args.skip_bids_validator:
    #process = run_cmd('bids-validator %s' % args.bids_dir)
    #logging.info(process.stdout)
    #if process.returncode != 0:
    #    logging.error('BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
    #                    'using the BIDS Validator (http://incf.github.io/bids-validator/).\nUse the '
    #                    '--skip_bids_validator argument to run the detection without prior BIDS validation.')
    #    exit(1)
    bids_error = False
    for dir_, d, files in os.walk(args.bids_dir):
        for file in files:
            rel_file = os.path.relpath(dir_, args.bids_dir)
            if rel_file[0] == '.':
                rel_file = rel_file[1:]
            rel_file = os.path.join(rel_file, file)
            if not BIDSValidator().is_bids('/' + rel_file):
                logging.error('Invalid BIDS-file: ' + rel_file)
                bids_error = True
    if bids_error:
        logging.error('BIDS input dataset did not pass BIDS validator. Datasets can be validated online '
                      'using the BIDS Validator (http://incf.github.io/bids-validator/).\nUse the '
                      '--skip_bids_validator argument to run the detection without prior BIDS validation.')
        exit(1)

#
# process per subject and subset
#

# make sure the output directory exists
if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        logging.error('Could not create output directory (\'' + args.output_dir + '\'), exiting...')
        exit(1)

# list the subject to analyze (either based on the input parameter or list all in the BIDS_dir)
subjects_to_analyze = []
if args.participant_label:
    # user-specified subjects
    subjects_to_analyze = args.participant_label

else:
    # all subjects
    subject_dirs = glob(os.path.join(args.bids_dir, 'sub-*'))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

#
for subject in subjects_to_analyze:

    # see if the subject is exists (in case the user specified the labels)
    if os.path.isdir(os.path.join(args.bids_dir, subject)):

        # retrieve the data formats to include
        if args.format_extension:
            extensions = args.format_extension
            for extension in extensions:
                if not any(extension in x for x in VALID_FORMAT_EXTENSIONS):
                    logging.error('Invalid data format extension \'' + extension + '\', exiting...')
                    exit(1)
        else:
            extensions = VALID_FORMAT_EXTENSIONS

        # build path patterns for the search of subsets
        subset_patterns = args.subset_search_pattern if args.subset_search_pattern else ('',)
        subsets = []
        modalities = ('*eeg',)                    # ieeg and eeg
        for extension in extensions:
            for modality in modalities:
                for subset_pattern in subset_patterns:
                    subsets += glob(os.path.join(args.bids_dir, subject, modality, '*' + subset_pattern + '*' + extension)) + \
                               glob(os.path.join(args.bids_dir, subject, '*', modality, '*' + subset_pattern + '*' + extension))

        # bring subsets with multiple formats down to one format (prioritized to occurrence in the extension var)
        for subset in subsets:
            subset_name = subset[:subset.rindex(".")]
            for subset_other in reversed(subsets):
                if not subset == subset_other:
                    subset_other_name = subset_other[:subset_other.rindex(".")]
                    if subset_name == subset_other_name:
                        subsets.remove(subset_other)

        # loop through the participant's subsets for analysis
        for subset in subsets:

            # print subset information
            logging.info('------------------------')
            log_indented_line('Subset:', subset)
            logging.info('')

            # derive the bids roots (subject/session and subset) from the full path
            bids_subjsess_root = os.path.commonprefix(glob(os.path.join(os.path.dirname(subset), '*.*')))[:-1]
            bids_subset_root = subset[:subset.rindex('_')]


            #
            # retrieve channel metadata
            #

            # retrieve the channel metadata from the channels.tsv file
            try:
                csv = load_channel_info(bids_subset_root + '_channels.tsv')
            except (FileNotFoundError, LookupError):
                logging.error('Could not load the channel metadata, exiting...')
                exit(1)

            # sort out the good, the bad and the... non-ieeg
            channels_bad = []                                       # channels excluded because they are marked as bad
            channels_incl_detect = []                               # the channel that are needed for detection
            channels_incl_early_reref = []                          # TODO: channel that are included for re-referencing
            channels_excl_detect_by_type = []
            channels_have_status = 'status' in csv.columns
            for index, row in csv.iterrows():
                excluded_for_detection = False

                # check if bad channel
                if channels_have_status:
                    if row['status'].lower() == 'bad':
                        channels_bad.append(row['name'])

                        # continue to the next channel
                        continue

                #
                if not row['type'].upper() in cfg('channels', 'types'):
                    channels_excl_detect_by_type.append(row['name'])
                    excluded_for_detection = True

                if not excluded_for_detection:
                    channels_incl_detect.append(row['name'])

            # print channel information
            logging.info(multi_line_list(channels_bad, LOGGING_CAPTION_INDENT_LENGTH, 'Bad channels (excluded):', 20, ' '))
            logging.info(multi_line_list(channels_excl_detect_by_type, LOGGING_CAPTION_INDENT_LENGTH, 'Channels excluded from detection by type:', 20, ' '))
            logging.info(multi_line_list(channels_incl_detect, LOGGING_CAPTION_INDENT_LENGTH, 'Channels included for detection:', 20, ' ', str(len(channels_incl_detect))))

            # check if there are any channels
            if len(channels_incl_detect) == 0:
                logging.error('No channels were found, exiting...')
                exit(1)
            logging.info('')


            #
            # retrieve trials
            #

            # retrieve the stimulation events (onsets and pairs) from the events.tsv file
            try:
                csv = load_event_info(bids_subset_root + '_events.tsv', ('trial_type', 'electrical_stimulation_site'))
            except (FileNotFoundError, LookupError):
                logging.error('Could not load the stimulation event metadata, exiting...')
                exit(1)

            # acquire the onset and electrode-pair for each stimulation
            trial_onsets = []
            trial_pairs = []
            trials_bad_onsets = []
            trials_have_status = 'status' in csv.columns
            for index, row in csv.iterrows():
                if row['trial_type'].lower() == 'electrical_stimulation':
                    if not is_number(row['onset']) or isnan(float(row['onset'])) or float(row['onset']) < 0:
                        logging.warning('Invalid onset \'' + row['onset'] + '\' in events, should be a numeric value >= 0. Discarding trial...')
                        continue

                    if trials_have_status:
                        if not row['status'].lower() == 'good':
                            trials_bad_onsets.append(row['onset'])
                            continue

                    pair = row['electrical_stimulation_site'].split('-')
                    if not len(pair) == 2 or len(pair[0]) == 0 or len(pair[1]) == 0:
                        logging.error('Electrical stimulation site \'' + row['electrical_stimulation_site'] + '\' invalid, should be two values separated by a dash (e.g. CH01-CH02), exiting...')
                        exit(1)
                    trial_onsets.append(float(row['onset']))
                    trial_pairs.append(pair)

            if len(trials_bad_onsets) > 0:
                log_indented_line('Number of trials marked as bad (excluded):', str(len(trials_bad_onsets)))

            # check if there are trials
            if len(trial_onsets) == 0:
                logging.error('No trials were found, exiting...')
                exit(1)


            #
            # retrieve stimulus-pairs
            #

            # determine the stimulation-pairs conditions (and the trial and electrodes that belong to them)
            # (note that the 'concat_bidirectional_pairs' configuration setting is taken into account here)
            #
            stim_pairs_onsets = dict()              # for each pair, the onsets of the trials that were involved
            stim_pairs_electrode_indices = dict()   # for each pair, the indices of the electrodes that were stimulated

            # loop over all the combinations of channels
            # Note:     only the combinations of stim-pairs that actually have events/trials end up in the output
            for iChannel0 in range(len(channels_incl_detect)):
                for iChannel1 in range(len(channels_incl_detect)):

                    # retrieve the indices of all the trials that concern this stim-pair
                    indices = []
                    if cfg('trials', 'concat_bidirectional_pairs'):
                        # allow concatenation of bidirectional pairs, pair order does not matter
                        if not iChannel1 < iChannel0:
                            # unique pairs while ignoring pair order
                            indices = [i for i, x in enumerate(trial_pairs) if
                                       (x[0] == channels_incl_detect[iChannel0] and x[1] == channels_incl_detect[iChannel1]) or (x[0] == channels_incl_detect[iChannel1] and x[1] == channels_incl_detect[iChannel0])]

                    else:
                        # do not concatenate bidirectional pairs, pair order matters
                        indices = [i for i, x in enumerate(trial_pairs) if
                                   x[0] == channels_incl_detect[iChannel0] and x[1] == channels_incl_detect[iChannel1]]

                    # add the pair if there are trials for it
                    if len(indices) > 0:
                        stim_pairs_onsets[channels_incl_detect[iChannel0] + '-' + channels_incl_detect[iChannel1]] = [trial_onsets[i] for i in indices]
                        stim_pairs_electrode_indices[channels_incl_detect[iChannel0] + '-' + channels_incl_detect[iChannel1]] = (iChannel0, iChannel1)

            # search for stimulus-pairs with too little trials
            stimpair_remove_indices = []
            for stim_pair, onsets in stim_pairs_onsets.items():
                if len(onsets) < cfg('trials', 'minimum_stimpair_trials'):
                    stimpair_remove_indices.append(stim_pair)

            # remove the stimulus-pairs with too little trials
            if len(stimpair_remove_indices) > 0:

                # message
                stimpair_print = [stim_pair + ' (' + str(len(stim_pairs_onsets[stim_pair])) + ' trials)' for stim_pair in stimpair_remove_indices]
                stimpair_print = [str_print.ljust(len(max(stimpair_print, key=len)), ' ') for str_print in stimpair_print]
                logging.info(multi_line_list(stimpair_print, LOGGING_CAPTION_INDENT_LENGTH, 'Stim-pairs excluded by number of trials:', 4, '   '))

                # remove those stimulation-pairs
                for stim_pair in stimpair_remove_indices:
                    del stim_pairs_onsets[stim_pair]
                    del stim_pairs_electrode_indices[stim_pair]

            # display stimulation-pair/trial information
            stimpair_print = [stim_pair + ' (' + str(len(onsets)) + ' trials)' for stim_pair, onsets in stim_pairs_onsets.items()]
            stimpair_print = [str_print.ljust(len(max(stimpair_print, key=len)), ' ') for str_print in stimpair_print]
            logging.info(multi_line_list(stimpair_print, LOGGING_CAPTION_INDENT_LENGTH, 'Stimulation pairs included:', 4, '   ', str(len(stim_pairs_onsets))))

            # check if there are stimulus-pairs
            if len(stim_pairs_onsets) == 0:
                logging.error('No stimulus-pairs were found, exiting...')
                exit(1)


            #
            # read and epoch the data
            #

            # prepare some preprocessing variables
            early_reref = None
            line_noise_removal = None
            late_reref = None

            if cfg('preprocess', 'early_re_referencing', 'enabled'):
                early_reref = RerefStruct.generate_car(channels_incl_detect)  # TODO: should be alle channels, not just channels_incl_detect
                # TODO: implement different re-referencing methods
                early_reref.set_exclude_reref_epochs(stim_pairs_onsets,
                                                     (cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[0], cfg('preprocess', 'early_re_referencing', 'stim_excl_epoch')[1]),
                                                     '-')

            if str(cfg('preprocess', 'line_noise_removal')).lower() == 'tsv':
                # TODO: implement from tsv, now just sets to 60
                line_noise_removal = 60
            if not str(cfg('preprocess', 'line_noise_removal')).lower() == 'off':
                line_noise_removal = float(cfg('preprocess', 'line_noise_removal'))

            #late_reref = RerefStruct.generate_car(channels_incl_detect)  # TODO: should be all channels, not just channels_incl_detect
            #late_reref.set_exclude_reref_epochs(stim_pairs_onsets, (-.01, 1.0), '-')


            # determine the metrics that should be produced
            metric_callbacks = tuple()
            if cfg('metrics', 'cross_proj', 'enabled'):
                metric_callbacks += tuple([metric_cross_proj])
            if cfg('metrics', 'waveform', 'enabled'):
                metric_callbacks += tuple([metric_waveform])

            # read, normalize, epoch and average the trials within the condition
            # Note: 'load_data_epochs_averages' is used instead of 'load_data_epochs' here because it is more memory
            #       efficient when only the averages are needed
            if len(metric_callbacks) == 0:
                logging.info('- Reading data...')
            else:
                logging.info('- Reading data and calculating metrics...')

            # TODO: normalize to raw or to Z-values (return both raw and z?)
            #       z-might be needed for detection
            try:
                sampling_rate, averages, metrics = load_data_epochs_averages(subset, channels_incl_detect, list(stim_pairs_onsets.values()),
                                                                             trial_epoch=cfg('trials', 'trial_epoch'),
                                                                             baseline_norm=cfg('trials', 'baseline_norm'),
                                                                             baseline_epoch=cfg('trials', 'baseline_epoch'),
                                                                             out_of_bound_handling=cfg('trials', 'out_of_bounds_handling'),
                                                                             metric_callbacks=metric_callbacks,
                                                                             high_pass=cfg('preprocess', 'high_pass'),
                                                                             early_reref=early_reref,
                                                                             line_noise_removal=line_noise_removal,
                                                                             late_reref=late_reref,
                                                                             preproc_priority=('speed' if preproc_prioritize_speed else 'mem'))
            except (ValueError, RuntimeError):
                logging.error('Could not load data (' + subset + '), exiting...')
                exit(1)

            # for each stimulation pair, NaN out the values of the electrodes that were stimulated
            # Note: the key order in stim_pairs_onsets and the first dimension of the CCEP averages matrix should match
            iPair = 0
            for stim_pair in stim_pairs_onsets.keys():
                averages[stim_pairs_electrode_indices[stim_pair][0], iPair, :] = np.nan
                averages[stim_pairs_electrode_indices[stim_pair][1], iPair, :] = np.nan
                iPair += 1

            # determine the sample of stimulus onset (counting from the epoch start)
            onset_sample = int(round(abs(cfg('trials', 'trial_epoch')[0] * sampling_rate)))
            # todo: handle trial epochs which start after the trial onset, currently disallowed by config

            # split out the metric results
            cross_proj_metrics = None
            waveform_metrics = None
            metric_counter = 0
            if cfg('metrics', 'cross_proj', 'enabled'):
                cross_proj_metrics = metrics[:, :, metric_counter]
                metric_counter += 1
            if cfg('metrics', 'waveform', 'enabled'):
                waveform_metrics = metrics[:, :, metric_counter]


            #
            # prepare an output directory
            #

            # make sure a subject directory exists
            output_root = os.path.join(args.output_dir, os.path.basename(os.path.normpath(bids_subset_root)))
            if not os.path.exists(output_root):
                try:
                    os.makedirs(output_root)
                except OSError as e:
                    logging.error("Could not create subset output directory (\'" + output_root + "\'), exiting...")
                    exit(1)

            # intermediate saving of the ccep data as .mat
            saveDict = dict()
            saveDict['sampling_rate'] = sampling_rate
            saveDict['onset_sample'] = onset_sample
            saveDict['ccep_average'] = averages
            saveDict['stimpair_labels'] = np.asarray(list(stim_pairs_onsets.keys()), dtype='object')
            saveDict['channel_labels'] = np.asarray(channels_incl_detect, dtype='object')
            saveDict['config'] = get_config_dict()
            if cfg('metrics', 'cross_proj', 'enabled'):
                saveDict['cross_proj_metrics'] = cross_proj_metrics
            if cfg('metrics', 'waveform', 'enabled'):
                saveDict['waveform_metrics'] = waveform_metrics
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'), saveDict)

            # write the configuration
            write_config(os.path.join(output_root, 'ccep_config.json'))


            #
            # perform the evoked response detection
            #

            # detect evoked responses
            logging.info('- Detecting evoked responses...')
            try:
                if cfg('detection', 'negative'):
                    er_neg_peak_indices, er_neg_peak_amplitudes = ieeg_detect_er(averages, onset_sample, int(sampling_rate),
                                                                                 cross_proj_metrics=cross_proj_metrics,
                                                                                 waveform_metrics=waveform_metrics)
                if cfg('detection', 'positive'):
                    er_pos_peak_indices, er_pos_peak_amplitudes = ieeg_detect_er(averages, onset_sample,
                                                                                 int(sampling_rate),
                                                                                 cross_proj_metrics=cross_proj_metrics,
                                                                                 waveform_metrics=waveform_metrics,
                                                                                 detect_positive=True)
            except (ValueError, RuntimeError):
                logging.error('Evoked response detection failed, exiting...')
                exit(1)

            # intermediate saving of the data and evoked response detection results as .mat
            if cfg('detection', 'negative'):
                saveDict['neg_peak_indices'] = er_neg_peak_indices
                saveDict['neg_peak_amplitudes'] = er_neg_peak_amplitudes
            if cfg('detection', 'positive'):
                saveDict['pos_peak_indices'] = er_pos_peak_indices
                saveDict['pos_peak_amplitudes'] = er_pos_peak_amplitudes
            sio.savemat(os.path.join(output_root, 'ccep_data.mat'), saveDict)


            #
            # generate images
            #

            if cfg('visualization', 'generate_electrode_images') or \
                cfg('visualization', 'generate_stimpair_images') or \
                cfg('visualization', 'generate_matrix_images'):

                #
                # prepare some settings for plotting
                #

                # generate the x-axis values
                # Note: TRIAL_EPOCH_START is not expected to start after the stimulus onset, currently disallowed by config
                x = np.arange(averages.shape[2])
                x = x / sampling_rate + cfg('trials', 'trial_epoch')[0]

                # determine the range on the x axis where the stimulus was in samples
                # Note: TRIAL_EPOCH_START is not expected to start after the stimulus onset, currently disallowed by config
                stim_start_x = int(round(abs(cfg('trials', 'trial_epoch')[0] - cfg('visualization', 'blank_stim_epoch')[0]) * sampling_rate)) - 1
                stim_end_x = stim_start_x + int(ceil(abs(cfg('visualization', 'blank_stim_epoch')[1] - cfg('visualization', 'blank_stim_epoch')[0]) * sampling_rate)) - 1

                # calculate the legend x position
                legend_x = cfg('visualization', 'x_axis_epoch')[1] - .13

                # determine the drawing properties
                plot_props = calc_sizes_and_fonts(OUTPUT_IMAGE_SIZE,
                                                  len(stim_pairs_onsets),
                                                  len(channels_incl_detect))

                #
                # generate the electrodes plot
                #
                if cfg('visualization', 'generate_electrode_images'):

                    #
                    logging.info('- Generating electrode plots...')

                    # create progress bar
                    print_progressbar(0, len(channels_incl_detect), prefix='Progress:', suffix='Complete', length=50)

                    # loop through electrodes
                    for iElec in range(len(channels_incl_detect)):

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_SIZE, plot_props['stimpair_y_image_height'], False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(channels_incl_detect[iElec] + '\n', fontsize=plot_props['title_font_size'], fontweight='bold')

                        # loop through the stimulation-pairs
                        for iPair in range(len(stim_pairs_onsets)):

                            # draw 0 line
                            y = np.empty((averages.shape[2], 1))
                            y.fill(len(stim_pairs_onsets) - iPair)
                            ax.plot(x, y, linewidth=plot_props['zero_line_thickness'], color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = averages[iElec, iPair, :] / 500
                            y += len(stim_pairs_onsets) - iPair

                            # nan out the stimulation
                            #TODO, only nan if within display range
                            y[stim_start_x:stim_end_x] = np.nan

                            # check if there is a signal to plot
                            if not np.isnan(y).all():

                                # plot the signal
                                ax.plot(x, y, linewidth=plot_props['signal_line_thickness'])

                                # if negative evoked potential is detected, plot it
                                if cfg('visualization', 'negative') and not isnan(er_neg_peak_indices[iElec, iPair]):
                                    xNeg = er_neg_peak_indices[iElec, iPair] / sampling_rate + cfg('trials', 'trial_epoch')[0]
                                    yNeg = er_neg_peak_amplitudes[iElec, iPair] / 500
                                    yNeg += len(stim_pairs_onsets) - iPair
                                    ax.plot(xNeg, yNeg, marker='o', color='blue')

                                # if positive evoked potential is detected, plot it
                                if cfg('visualization', 'positive') and not isnan(er_pos_peak_indices[iElec, iPair]):
                                    xPos = er_pos_peak_indices[iElec, iPair] / sampling_rate + cfg('trials', 'trial_epoch')[0]
                                    yPos = er_pos_peak_amplitudes[iElec, iPair] / 500
                                    yPos += len(stim_pairs_onsets) - iPair
                                    ax.plot(xPos, yPos, marker='o', color=(0, 0, .6))

                        # set the x-axis
                        ax.set_xlabel('\ntime (s)', fontsize=plot_props['axis_label_font_size'])
                        ax.set_xlim(cfg('visualization', 'x_axis_epoch'))
                        for label in ax.get_xticklabels():
                            label.set_fontsize(plot_props['axis_ticks_font_size'])

                        # set the y-axis
                        ax.set_ylabel('Stimulated electrode-pair\n', fontsize=plot_props['axis_label_font_size'])
                        ax.set_ylim((0, len(stim_pairs_onsets) + 1))
                        ax.set_yticks(np.arange(1, len(stim_pairs_onsets) + 1, 1))
                        ax.set_yticklabels(np.flip(list(stim_pairs_onsets.keys())), fontsize=plot_props['stimpair_axis_ticks_font_size'])
                        ax.spines['bottom'].set_linewidth(1.5)
                        ax.spines['left'].set_linewidth(1.5)

                        # draw legend
                        legend_y = 2 if len(stim_pairs_onsets) > 4 else (1 if len(stim_pairs_onsets) > 1 else 0)
                        ax.plot([legend_x, legend_x], [legend_y + .05, legend_y + .95], linewidth=plot_props['legend_line_thickness'], color=(0, 0, 0))
                        ax.text(legend_x + .01, legend_y + .3, '500 \u03bcV', fontsize=plot_props['legend_font_size'])

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # save figure
                        fig.savefig(os.path.join(output_root, 'electrode_' + str(channels_incl_detect[iElec]) + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iElec + 1, len(channels_incl_detect), prefix='Progress:', suffix='Complete', length=50)

                #
                # generate the stimulation-pair plots
                #
                if cfg('visualization', 'generate_stimpair_images'):

                    #
                    logging.info('- Generating stimulation-pair plots...')

                    # create progress bar
                    print_progressbar(0, len(stim_pairs_onsets), prefix='Progress:', suffix='Complete', length=50)

                    # loop through the stimulation-pairs
                    # Note: the key order in stim_pairs_onsets and the first dimension of the CCEP averages matrix should match
                    iPair = 0
                    for stim_pair in stim_pairs_onsets.keys():

                        # create a figure and retrieve the axis
                        fig = create_figure(OUTPUT_IMAGE_SIZE, plot_props['electrode_y_image_height'], False)
                        ax = fig.gca()

                        # set the title
                        ax.set_title(stim_pair + '\n', fontsize=plot_props['title_font_size'], fontweight='bold')

                        # loop through the electrodes
                        for iElec in range(len(channels_incl_detect)):

                            # draw 0 line
                            y = np.empty((averages.shape[2], 1))
                            y.fill(len(channels_incl_detect) - iElec)
                            ax.plot(x, y, linewidth=plot_props['zero_line_thickness'], color=(0.8, 0.8, 0.8))

                            # retrieve the signal
                            y = averages[iElec, iPair, :] / 500
                            y += len(channels_incl_detect) - iElec

                            # nan out the stimulation
                            #TODO, only nan if within display range
                            y[stim_start_x:stim_end_x] = np.nan

                            # plot the signal
                            ax.plot(x, y, linewidth=plot_props['signal_line_thickness'])

                            # if evoked potential is detected, plot it
                            if cfg('visualization', 'negative') and not isnan(er_neg_peak_indices[iElec, iPair]):
                                xNeg = er_neg_peak_indices[iElec, iPair] / sampling_rate + cfg('trials', 'trial_epoch')[0]
                                yNeg = er_neg_peak_amplitudes[iElec, iPair] / 500
                                yNeg += len(channels_incl_detect) - iElec
                                ax.plot(xNeg, yNeg, marker='o', color='blue')

                            if cfg('visualization', 'positive') and not isnan(er_pos_peak_indices[iElec, iPair]):
                                xPos = er_pos_peak_indices[iElec, iPair] / sampling_rate + cfg('trials', 'trial_epoch')[0]
                                yPos = er_pos_peak_amplitudes[iElec, iPair] / 500
                                yPos += len(channels_incl_detect) - iElec
                                ax.plot(xPos, yPos, marker='o', color=(0, 0, .6))

                        # set the x-axis
                        ax.set_xlabel('\ntime (s)', fontsize=plot_props['axis_label_font_size'])
                        ax.set_xlim(cfg('visualization', 'x_axis_epoch'))
                        for label in ax.get_xticklabels():
                            label.set_fontsize(plot_props['axis_ticks_font_size'])

                        # set the y-axis
                        ax.set_ylabel('Measured electrodes\n', fontsize=plot_props['axis_label_font_size'])
                        ax.set_ylim((0, len(channels_incl_detect) + 1))
                        ax.set_yticks(np.arange(1, len(channels_incl_detect) + 1, 1))
                        ax.set_yticklabels(np.flip(channels_incl_detect), fontsize=plot_props['electrode_axis_ticks_font_size'])
                        ax.spines['bottom'].set_linewidth(1.5)
                        ax.spines['left'].set_linewidth(1.5)

                        # draw legend
                        legend_y = 2 if len(stim_pairs_onsets) > 4 else (1 if len(stim_pairs_onsets) > 1 else 0)
                        ax.plot([legend_x, legend_x], [legend_y + .05, legend_y + .95], linewidth=plot_props['legend_line_thickness'], color=(0, 0, 0))
                        ax.text(legend_x + .01, legend_y + .3, '500 \u03bcV', fontsize=plot_props['legend_font_size'])

                        # Hide the right and top spines
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        # save figure
                        fig.savefig(os.path.join(output_root, 'stimpair_' + stim_pair + '.png'), bbox_inches='tight')

                        # update progress bar
                        print_progressbar(iPair + 1, len(stim_pairs_onsets), prefix='Progress:', suffix='Complete', length=50)

                        #
                        iPair += 1


                #
                # generate the matrices
                #
                if cfg('visualization', 'generate_matrix_images'):

                    #
                    logging.info('- Generating matrices...')

                    image_width, image_height = calc_matrix_image_size(plot_props['stimpair_y_image_height'],
                                                                       len(stim_pairs_onsets),
                                                                       len(channels_incl_detect))

                    # generate negative matrices and save
                    if cfg('visualization', 'negative'):

                        # amplitude
                        fig = gen_amplitude_matrix(list(stim_pairs_onsets.keys()), channels_incl_detect,
                                                   plot_props, image_width, image_height,
                                                   er_neg_peak_amplitudes.copy() * -1, False)
                        fig.savefig(os.path.join(output_root, 'matrix_amplitude_neg.png'), bbox_inches='tight')

                        # latency
                        fig = gen_latency_matrix(list(stim_pairs_onsets.keys()), channels_incl_detect,
                                                 plot_props, image_width, image_height,
                                                 (er_neg_peak_indices.copy() - onset_sample) / sampling_rate * 1000)     # convert the indices (in samples) to time units (ms)
                        fig.savefig(os.path.join(output_root, 'matrix_latency_neg.png'), bbox_inches='tight')

                    # generate positive matrices and save
                    if cfg('visualization', 'positive'):

                        # amplitude
                        fig = gen_amplitude_matrix(list(stim_pairs_onsets.keys()), channels_incl_detect,
                                                   plot_props, image_width, image_height,
                                                   er_pos_peak_amplitudes.copy(), True)
                        fig.savefig(os.path.join(output_root, 'matrix_amplitude_pos.png'), bbox_inches='tight')

                        # latency
                        fig = gen_latency_matrix(list(stim_pairs_onsets.keys()), channels_incl_detect,
                                                 plot_props, image_width, image_height,
                                                 (er_pos_peak_indices.copy() - onset_sample) / sampling_rate * 1000)     # convert the indices (in samples) to time units (ms)
                        fig.savefig(os.path.join(output_root, 'matrix_latency_pos.png'), bbox_inches='tight')

            #
            logging.info('- Finished subset')

    else:
        #
        logging.warning('Participant \'' + subject + '\' could not be found, skipping')


logging.info('- Finished running')
