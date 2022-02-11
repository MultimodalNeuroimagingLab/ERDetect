"""
Module to store and manage the application's configuration variables
Note: In python, each module becomes a single global instance and is therefore technically a singleton


Copyright 2022, Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import json
from utils.misc import is_number, is_valid_numeric_range, numbers_to_padded_string


#
# constants
#
VALID_FORMAT_EXTENSIONS         = ('.mefd', '.edf', '.vhdr', '.vmrk', '.eeg')   # valid data format to search for (European Data Format, BrainVision and MEF3)
OUTPUT_IMAGE_SIZE               = 2000                                          # the number of pixels that is used as the "initial" height or width for the output images
LOGGING_CAPTION_INDENT_LENGTH   = 50                                            # the indent length of the caption in a logging output string
VALID_CHANNEL_TYPES             = ('EEG', 'ECOG', 'SEEG', 'DBS', 'VEOG', 'HEOG', 'EOG', 'ECG', 'EMG', 'TRIG', 'AUDIO', 'PD', 'EYEGAZE', 'PUPIL', 'MISC', 'SYSCLOCK', 'ADC', 'DAC', 'REF', 'OTHER')


def __create_default_config():
    """
    Create and return a config dictionary with default values

    Returns:
        config (dict):                        A config dictionary with default values
    """

    config = dict()
    config['trials'] = dict()
    config['trials']['trial_epoch']                         = (-1.0, 2.0)           # the time-span (in seconds) relative to the stimulus onset that will be used to extract the signal for each trial
    config['trials']['out_of_bounds_handling']              = 'first_last_only'     #
    # TODO: for comparison now set to -.1s. Should check metric results in matlab if I change those to -.02
    #config['trials']['baseline_epoch']                      = (-0.5, -0.02)  # the time-span (in seconds) relative to the stimulus onset that will be considered as the start and end of the baseline epoch within each trial
    config['trials']['baseline_epoch']                      = (-0.5, -0.1)         # the time-span (in seconds) relative to the stimulus onset that will be considered as the start and end of the baseline epoch within each trial
    config['trials']['baseline_norm']                       = 'median'
    config['trials']['concat_bidirectional_pairs']          = True                  # concatenate electrode pairs that were stimulated in both directions (e.g. CH01-CH02 and CH02-CH01)
    config['trials']['minimum_stimpair_trials']             = 5                     # the minimum number of stimulation trials that are needed for a stimulus-pair to be included

    config['channels'] = dict()
    config['channels']['types']                             = ('ECOG', 'SEEG', 'DBS')

    config['cross_projection_metric'] = dict()
    config['cross_projection_metric']['enabled']            = True
    config['cross_projection_metric']['epoch']              = (0.012, 0.09)

    config['waveform_metric'] = dict()
    config['waveform_metric']['enabled']                    = True
    config['waveform_metric']['epoch']                      = (0.012, 0.09)
    config['waveform_metric']['bandpass']                   = (10, 30)

    config['n1_detect'] = dict()
    config['n1_detect']['peak_search_epoch']                = (0, 0.5)
    config['n1_detect']['n1_search_epoch']                  = (0.009, 0.09)
    config['n1_detect']['n1_baseline_epoch']                = (-1, -0.1)
    config['n1_detect']['n1_baseline_threshold_factor']     = 3.4

    config['visualization'] = dict()
    config['visualization']['x_axis_epoch']                 = (-0.2, 1)             # the range for the x-axis in display, (in seconds) relative to the stimulus onset that will be used as the range
    config['visualization']['blank_stim_epoch']             = (-0.015, 0.0025)      # the range
    config['visualization']['generate_electrode_images']    = True
    config['visualization']['generate_stimpair_images']     = True
    config['visualization']['generate_matrix_images']       = True

    # return a default configuration
    return config


def get(level1, level2):
    """
    Retrieve a configuration value

    Args:
        level1 (str):                         The first level in the configuration structure
        level2 (str):                         The second level in the configuration structure

    Returns:
        The configuration value within the structure
    """
    global _config
    return _config[level1][level2]

def get_config_dict():
    """
    Retrieve the configuration dictionary

    Returns:
        The configuration dictionary
    """
    return _config

def load_config(filepath):
    """
    Load and set the configuration based on a configuration file

    Args:
        filepath (str):                       The path to the configuration file to load

    Returns:
        bool:                                 True for success, False otherwise
    """

    # first retrieve a default config
    config = __create_default_config()

    # try to read the JSON configuration file
    try:
        with open(filepath) as json_file:
            json_config = json.load(json_file)
    except IOError:
        logging.error('Could not access configuration file at \'' + filepath + '\'')
        return None
    except json.decoder.JSONDecodeError as e:
        logging.error('Could not interpret configuration file at \'' + filepath + '\', make sure the JSON syntax is valid: \'' + str(e) + '\'')
        return None

    #
    # read helper functions
    #

    def retrieve_config_bool(json_dict, ref_config, level1, level2):
        if level1 in json_dict:
            if level2 in json_dict[level1]:
                try:
                    ref_config[level1][level2] = bool(json_dict[level1][level2])
                except:
                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a boolean (true, false, 0 or 1)')
                    return False
        return True

    def retrieve_config_number(json_dict, ref_config, level1, level2):
        if level1 in json_dict:
            if level2 in json_dict[level1]:
                if is_number(json_dict[level1][level2]):
                    ref_config[level1][level2] = float(json_dict[level1][level2])
                else:
                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a single number')
                    return False
        return True

    def retrieve_config_range(json_dict, ref_config, level1, level2):
        if level1 in json_dict:
            if level2 in json_dict[level1]:
                if is_valid_numeric_range(json_dict[level1][level2]):
                    ref_config[level1][level2] = (json_dict[level1][level2][0], json_dict[level1][level2][1])
                else:
                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be an array of two numbers')
                    return False
        return True

    def retrieve_config_string(json_dict, ref_config, level1, level2, options=None, case_sensitive=False):
        if level1 in json_dict:
            if level2 in json_dict[level1]:
                if isinstance(json_dict[level1][level2], str):
                    if options is None:
                        ref_config[level1][level2] = json_dict[level1][level2]
                    else:
                        value_cased = json_dict[level1][level2]
                        if not case_sensitive:
                            options = (option.lower() for option in options)
                            value_cased = value_cased.lower()
                        if value_cased in options:
                            ref_config[level1][level2] = json_dict[level1][level2]
                        else:
                            logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value can only be one of the following options: ' + str(options)[1:-1])
                            return False
                else:
                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a string')
                    return False
        return True

    def retrieve_config_tuple(json_dict, ref_config, level1, level2, options=None, case_sensitive=False):
        if level1 in json_dict:
            if level2 in json_dict[level1]:
                if isinstance(json_dict[level1][level2], list):
                    if options is None:
                        ref_config[level1][level2] = tuple(json_dict[level1][level2])
                    else:
                        options_cased = options
                        values_cased = json_dict[level1][level2]
                        if not case_sensitive:
                            options_cased = [option.lower() for option in options]
                            values_cased = [value.lower() for value in values_cased]
                        ref_config[level1][level2] = list()
                        for value in values_cased:
                            if value in options_cased:
                                ref_config[level1][level2].append(value)
                            else:
                                logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the following values are allowed: ' + str(options)[1:-1])
                                return False
                        ref_config[level1][level2] = tuple(ref_config[level1][level2])
                else:
                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should an array of strings')
                    return False
        return True

    #
    # retrieve the settings
    #

    # trials settings
    if not retrieve_config_range(json_config, config, 'trials', 'trial_epoch'):
        return False
    if not retrieve_config_string(json_config, config, 'trials', 'out_of_bounds_handling', ('error', 'first_last_only', 'allow')):
        return False
    config['trials']['out_of_bounds_handling'] = str(config['trials']['out_of_bounds_handling']).lower()
    if not retrieve_config_range(json_config, config, 'trials', 'baseline_epoch'):
        return False
    if not retrieve_config_string(json_config, config, 'trials', 'baseline_norm', ('median', 'mean', 'none')):
        return False
    config['trials']['baseline_norm'] = str(config['trials']['baseline_norm']).lower()
    if not retrieve_config_bool(json_config, config, 'trials', 'concat_bidirectional_pairs'):
        return False
    if not retrieve_config_number(json_config, config, 'trials', 'minimum_stimpair_trials'):
        return False
    if not config['trials']['minimum_stimpair_trials'] == round(config['trials']['minimum_stimpair_trials']):
        logging.error('Invalid value in the configuration file for trials->minimum_stimpair_trials, the value should be an integer')
        return False
    if config['trials']['minimum_stimpair_trials'] < 0:
        logging.error('Invalid value in the configuration file for trials->minimum_stimpair_trials, the value can be 0 (no trial limit) or higher')
        return False
    config['trials']['minimum_stimpair_trials'] = int(config['trials']['minimum_stimpair_trials'])

    # channel settings
    if not retrieve_config_tuple(json_config, config, 'channels', 'types', VALID_CHANNEL_TYPES):
        return False
    if len(config['channels']['types']) == 0:
        logging.error('Invalid value in the configuration file for channels->types, at least one channel type should be given')
        return False
    config['channels']['types'] = [value.upper() for value in config['channels']['types']]

    # cross-projection metric settings
    if not retrieve_config_bool(json_config, config, 'cross_projection_metric', 'enabled'):
        return False
    if not retrieve_config_range(json_config, config, 'cross_projection_metric', 'epoch'):
        return False

    # waveform metric settings
    if not retrieve_config_bool(json_config, config, 'waveform_metric', 'enabled'):
        return False
    if not retrieve_config_range(json_config, config, 'waveform_metric', 'epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'waveform_metric', 'bandpass'):
        return False

    # n1 app settings
    if not retrieve_config_range(json_config, config, 'n1_detect', 'peak_search_epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'n1_detect', 'n1_search_epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'n1_detect', 'n1_baseline_epoch'):
        return False
    if not retrieve_config_number(json_config, config, 'n1_detect', 'n1_baseline_threshold_factor'):
        return False

    # visualization settings
    if not retrieve_config_range(json_config, config, 'visualization', 'x_axis_epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'visualization', 'blank_stim_epoch'):
        return False
    if not retrieve_config_bool(json_config, config, 'visualization', 'generate_electrode_images'):
        return False
    if not retrieve_config_bool(json_config, config, 'visualization', 'generate_stimpair_images'):
        return False
    if not retrieve_config_bool(json_config, config, 'visualization', 'generate_matrix_images'):
        return False

    # perform sanity checks on the loaded configuration values
    if not __check_config(config):
        logging.error('Invalid configuration...')
        return False

    # replace the current config dictionary with the loaded dictionary
    global _config
    _config = config

    # return success
    return True


def write_config(filepath):
    """
    Write the current configuration to a configuration file

    Args:
        filepath (str):                       The path write the configuration file to
    """
    global _config

    # save the configuration that was used
    config_str = '{\n    "trials": {\n' \
                '        "trial_epoch":                      [' + numbers_to_padded_string(_config['trials']['trial_epoch'], 16) + '],\n' \
                '        "out_of_bounds_handling":           "' + _config['trials']['out_of_bounds_handling'] + '",\n' \
                '        "baseline_epoch":                   [' + numbers_to_padded_string(_config['trials']['baseline_epoch'], 16) + '],\n' \
                '        "baseline_norm":                    "' + _config['trials']['baseline_norm'] + '",\n' \
                '        "concat_bidirectional_pairs":       ' + ('true' if _config['trials']['concat_bidirectional_pairs'] else 'false') + ',\n' \
                '        "minimum_stimpair_trials":          ' + str(_config['trials']['minimum_stimpair_trials']) + '\n' \
                '    },\n\n' \
                '    "channels": {\n' \
                '        "types":                            ' + json.dumps(_config['channels']['types']) + '\n' \
                '    },\n\n' \
                '    "cross_projection_metric": {\n' \
                '        "enabled":                          ' + json.dumps(_config['cross_projection_metric']['enabled']) + ',\n' \
                '        "epoch":                            [' + numbers_to_padded_string(_config['cross_projection_metric']['epoch'], 16) + ']\n' \
                '    },\n\n' \
                '    "waveform_metric": {\n' \
                '        "enabled":                          ' + json.dumps(_config['waveform_metric']['enabled']) + ',\n' \
                '        "epoch":                            [' + numbers_to_padded_string(_config['waveform_metric']['epoch'], 16) + '],\n' \
                '        "bandpass":                         [' + numbers_to_padded_string(_config['waveform_metric']['bandpass'], 16) + ']\n' \
                '    },\n\n' \
                '    "n1_detect": {\n' \
                '        "peak_search_epoch":                [' + numbers_to_padded_string(_config['n1_detect']['peak_search_epoch'], 16) + '],\n' \
                '        "n1_search_epoch":                  [' + numbers_to_padded_string(_config['n1_detect']['n1_search_epoch'], 16) + '],\n' \
                '        "n1_baseline_epoch":                [' + numbers_to_padded_string(_config['n1_detect']['n1_baseline_epoch'], 16) + '],\n' \
                '        "n1_baseline_threshold_factor":     ' + str(_config['n1_detect']['n1_baseline_threshold_factor']) + '\n' \
                '    },\n\n' \
                '    "visualization": {\n' \
                '        "x_axis_epoch":                     [' + numbers_to_padded_string(_config['visualization']['x_axis_epoch'], 16) + '],\n' \
                '        "blank_stim_epoch":                 [' + numbers_to_padded_string(_config['visualization']['blank_stim_epoch'], 16) + '],\n' \
                '        "generate_electrode_images":        ' + ('true' if _config['visualization']['generate_electrode_images'] else 'false') + ',\n' \
                '        "generate_stimpair_images":         ' + ('true' if _config['visualization']['generate_stimpair_images'] else 'false') + ',\n' \
                '        "generate_matrix_images":           ' + ('true' if _config['visualization']['generate_matrix_images'] else 'false') + '\n' \
                '    }\n' \
                '}'

    with open(filepath, 'w') as json_out:
        json_out.write(config_str + '\n')
        json_out.close()


def __check_config(config):
    """
    Perform sanity chech on a given configuration

    Args:
        config (dict):                        The configuration to check

    Returns:
        bool:                                 True when passing all checks, False elsewise
    """
    def check_epoch_start_after_onset(ref_config, level1, level2):
        if ref_config[level1][level2][0] < 0:
            logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the epoch should start after the stimulus onset (>= 0s)')
            return False
        return True

    def check_epoch_within_trial(ref_config, level1, level2):
        if ref_config[level1][level2][0] < config['trials']['trial_epoch'][0]:
            logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given start-point (at ' + str(ref_config[level1][level2][0]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
            return False
        if ref_config[level1][level2][1] > config['trials']['trial_epoch'][1]:
            logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][1]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
            return False
        return True

    def check_range_order(ref_config, level1, level2):
        if ref_config[level1][level2][1] < ref_config[level1][level2][0]:
            logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][1]) + 's) lies before the start-point (at ' + str(ref_config[level1][level2][0]) + 's)')
            return False
        if ref_config[level1][level2][0] == ref_config[level1][level2][1]:
            logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given start and end-point are the same (' + str(ref_config[level1][level2][0]) + 's)')
            return False
        return True

    # parameter start-end order
    if not check_range_order(config, 'trials', 'trial_epoch'):
        return False
    if not check_range_order(config, 'trials', 'baseline_epoch'):
        return False
    if not check_range_order(config, 'cross_projection_metric', 'epoch'):
        return False
    if not check_range_order(config, 'waveform_metric', 'epoch'):
        return False
    if not check_range_order(config, 'n1_detect', 'peak_search_epoch'):
        return False
    if not check_range_order(config, 'n1_detect', 'n1_search_epoch'):
        return False
    if not check_range_order(config, 'n1_detect', 'n1_baseline_epoch'):
        return False
    if not check_range_order(config, 'visualization', 'x_axis_epoch'):
        return False
    if not check_range_order(config, 'visualization', 'blank_stim_epoch'):
        return False

    # app epoch parameters should be within trial epoch
    if not check_epoch_within_trial(config, 'cross_projection_metric', 'epoch'):
        return False
    if not check_epoch_within_trial(config, 'waveform_metric', 'epoch'):
        return False
    if not check_epoch_within_trial(config, 'n1_detect', 'peak_search_epoch'):
        return False
    if not check_epoch_within_trial(config, 'n1_detect', 'n1_search_epoch'):
        return False
    if not check_epoch_within_trial(config, 'n1_detect', 'n1_baseline_epoch'):
        return False
    if not check_epoch_within_trial(config, 'visualization', 'x_axis_epoch'):
        return False
    if not check_epoch_within_trial(config, 'visualization', 'blank_stim_epoch'):
        return False

    # trial epoch show start before the stimulus onset (routines in run rely on that)
    if not check_epoch_start_after_onset(config, 'trials', 'trial_epoch'):
        return False

    # metric epochs should be after stimulus onset
    if not check_epoch_start_after_onset(config, 'cross_projection_metric', 'epoch'):
        return False
    if not check_epoch_start_after_onset(config, 'waveform_metric', 'epoch'):
        return False

    # app peak search should be after stimulus onset
    if not check_epoch_start_after_onset(config, 'n1_detect', 'peak_search_epoch'):
        return False
    if not check_epoch_start_after_onset(config, 'n1_detect', 'n1_search_epoch'):
        return False

    # the baseline threshold factor should be a positive number
    if config['n1_detect']['n1_baseline_threshold_factor'] <= 0:
        logging.error('Invalid [\'n1_detect\'][\'n1_baseline_threshold_factor\'] parameter, the threshold should be a positive value (> 0)')
        return False

    # the waveform bandpass limits
    if config['waveform_metric']['bandpass'][0] <= 0:
        logging.error('Invalid [\'waveform_metric\'][\'bandpass\'] parameter, the given lower cutoff frequency should be a positive number (' + str(config['waveform_metric']['bandpass'][0]) + ')')
        return False
    if config['waveform_metric']['bandpass'][1] <= 0:
        logging.error('Invalid [\'waveform_metric\'][\'bandpass\'] parameter, the given upper cutoff frequency should be a positive number (' + str(config['waveform_metric']['bandpass'][1]) + ')')
        return False
    if config['waveform_metric']['bandpass'][1] < config['waveform_metric']['bandpass'][0]:
        logging.error('Invalid [\'waveform_metric\'][\'bandpass\'] parameter, the upper cutoff frequency (' + str(config['waveform_metric']['bandpass'][1]) + ') is smaller than the lower cutoff frequency (' + str(config['waveform_metric']['bandpass'][0]) + ')')
        return False
    if config['waveform_metric']['bandpass'][0] == config['waveform_metric']['bandpass'][1]:
        logging.error('Invalid [\'waveform_metric\'][\'bandpass\'] parameter, the given lower and upper cutoff frequencies are the same (' + str(config['waveform_metric']['bandpass'][0]) + ')')
        return False

    # return success
    return True


# initialize a variable with a default configuration dictionary for this module
_config = __create_default_config()
