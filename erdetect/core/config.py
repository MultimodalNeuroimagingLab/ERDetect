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
from erdetect.utils.misc import is_number, is_valid_numeric_range, numbers_to_padded_string


#
# constants
#
OUTPUT_IMAGE_SIZE               = 2000                                          # the number of pixels that is used as the "initial" height or width for the output images
LOGGING_CAPTION_INDENT_LENGTH   = 50                                            # the indent length of the caption in a logging output string
VALID_CHANNEL_TYPES             = ('EEG', 'ECOG', 'SEEG', 'DBS', 'VEOG', 'HEOG', 'EOG', 'ECG', 'EMG', 'TRIG', 'AUDIO', 'PD', 'EYEGAZE', 'PUPIL', 'MISC', 'SYSCLOCK', 'ADC', 'DAC', 'REF', 'OTHER')

CONFIG_DETECTION_STD_BASE_BASELINE_EPOCH_DEFAULT = (-1, -0.1)
CONFIG_DETECTION_STD_BASE_BASELINE_THRESHOLD_FACTOR = 3.4
CONFIG_DETECTION_CROSS_PROJ_THRESHOLD = 3.5
CONFIG_DETECTION_WAVEFORM_PROJ_THRESHOLD = 1000


def create_default_config():
    """
    Create and return a config dictionary with default values

    Returns:
        config (dict):                        A config dictionary with default values
    """

    config = dict()

    config['preprocess'] = dict()
    config['preprocess']['high_pass']                               = False                     #
    config['preprocess']['early_re_referencing'] = dict()
    config['preprocess']['early_re_referencing']['enabled']         = False                     #
    config['preprocess']['early_re_referencing']['method']          = 'CAR'                     #
    config['preprocess']['early_re_referencing']['stim_excl_epoch'] = (-1.0, 2.0)
    config['preprocess']['early_re_referencing']['channel_types']   = ('ECOG', 'SEEG', 'DBS')   # the type of channels that will be included for early re-referencing
    config['preprocess']['line_noise_removal']                      = 'off'                     # TODO: off, json, 60, 50, 60Hz 50Hz
    config['preprocess']['late_re_referencing'] = dict()
    config['preprocess']['late_re_referencing']['enabled']          = False                     #
    config['preprocess']['late_re_referencing']['method']           = 'CAR'                     #
    config['preprocess']['late_re_referencing']['stim_excl_epoch']  = (-1.0, 2.0)
    config['preprocess']['late_re_referencing']['channel_types']    = ('ECOG', 'SEEG', 'DBS')   # the type of channels that will be included for late re-referencing

    config['trials'] = dict()
    config['trials']['trial_epoch']                                 = (-1.0, 2.0)               # the time-span (in seconds) relative to the stimulus onset that will be used to extract the signal for each trial
    config['trials']['out_of_bounds_handling']                      = 'first_last_only'         #
    config['trials']['baseline_epoch']                             = (-0.5, -0.02)              # the time-span (in seconds) relative to the stimulus onset that will be considered as the start and end of the baseline epoch within each trial
    config['trials']['baseline_norm']                               = 'median'
    config['trials']['concat_bidirectional_pairs']                  = True                      # concatenate electrode pairs that were stimulated in both directions (e.g. CH01-CH02 and CH02-CH01)
    config['trials']['minimum_stimpair_trials']                     = 5                         # the minimum number of stimulation trials that are needed for a stimulus-pair to be included

    config['channels'] = dict()
    config['channels']['measured_types']                            = ('ECOG', 'SEEG', 'DBS')   # the type of channels that will be included as measured electrodes
    config['channels']['stim_types']                                = ('ECOG', 'SEEG', 'DBS')   # the type of channels that are allowed to be included as part of the stim-pairs

    config['metrics'] = dict()
    config['metrics']['cross_proj'] = dict()
    config['metrics']['cross_proj']['enabled']                      = True
    config['metrics']['cross_proj']['epoch']                        = (0.012, 0.09)
    config['metrics']['waveform'] = dict()
    config['metrics']['waveform']['enabled']                        = True
    config['metrics']['waveform']['epoch']                          = (0.012, 0.09)
    config['metrics']['waveform']['bandpass']                       = (10, 30)

    config['detection'] = dict()
    config['detection']['negative']                                 = True                      # whether to detect negative responses
    config['detection']['positive']                                 = False                     # whether to detect positive responses
    config['detection']['peak_search_epoch']                        = (0, 0.5)
    config['detection']['response_search_epoch']                    = (0.009, 0.09)
    config['detection']['method']                                   = 'std_base'
    config['detection']['std_base'] = dict()
    config['detection']['std_base']['baseline_epoch']               = CONFIG_DETECTION_STD_BASE_BASELINE_EPOCH_DEFAULT
    config['detection']['std_base']['baseline_threshold_factor']    = CONFIG_DETECTION_STD_BASE_BASELINE_THRESHOLD_FACTOR

    config['visualization'] = dict()
    config['visualization']['negative']                             = True                      # whether to output negative responses
    config['visualization']['positive']                             = False                     # whether to output positive responses
    config['visualization']['x_axis_epoch']                         = (-0.2, 1)                 # the range for the x-axis in display, (in seconds) relative to the stimulus onset that will be used as the range
    config['visualization']['blank_stim_epoch']                     = (-0.015, 0.0025)          # the range
    config['visualization']['generate_electrode_images']            = True
    config['visualization']['generate_stimpair_images']             = True
    config['visualization']['generate_matrix_images']               = True

    # return a default configuration
    return config


def get(level1, level2, level3=None):
    """
    Retrieve a configuration value

    Args:
        level1 (str):                         The first level in the configuration structure
        level2 (str):                         The second level in the configuration structure
        level3 (str, optional):               The third level in the configuration structure

    Returns:
        The configuration value within the structure
    """
    global _config
    if level3 is None:
        return _config[level1][level2]
    else:
        return _config[level1][level2][level3]


def get_config_dict():
    """
    Retrieve the configuration dictionary

    Returns:
        The configuration dictionary
    """
    return _config


def set(value, level1, level2, level3=None):
    """
    Set a configuration value

    Args:
        level1 (str):                         The first level in the configuration structure
        level2 (str):                         The second level in the configuration structure
        level3 (str, optional):               The third level in the configuration structure

    """
    global _config

    if level1 not in _config:
        _config[level1] = dict()

    if level3 is None:
        _config[level1][level2] = value
    else:

        if not level2 in _config[level1]:
            _config[level1][level2] = dict()

        _config[level1][level2][level3] = value


def rem(level1, level2, level3=None):
    """
    Remove a configuration value

    Args:
        level1 (str):                         The first level in the configuration structure
        level2 (str):                         The second level in the configuration structure
        level3 (str, optional):               The third level in the configuration structure

    """
    global _config
    if level3 is None:
        _config[level1].pop(level2, None)
    else:
        _config[level1][level2].pop(level3, None)


def load_config(filepath):
    """
    Load and set the configuration based on a configuration file

    Args:
        filepath (str):                       The path to the configuration file to load

    Returns:
        bool:                                 True for success, False otherwise
    """

    # first retrieve a default config
    config = create_default_config()

    # try to read the JSON configuration file
    try:
        with open(filepath) as json_file:
            json_config = json.load(json_file)
    except IOError:
        logging.error('Could not access configuration file at \'' + filepath + '\'')
        return False
    except json.decoder.JSONDecodeError as e:
        logging.error('Could not interpret configuration file at \'' + filepath + '\', make sure the JSON syntax is valid: \'' + str(e) + '\'')
        return False

    #
    # read helper functions
    #

    def retrieve_config_bool(json_dict, ref_config, level1, level2, level3=None):
        if level1 in json_dict:
            if level2 in json_dict[level1]:

                if level3 is None:
                    try:
                        ref_config[level1][level2] = bool(json_dict[level1][level2])
                    except:
                        logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a boolean (true, false, 0 or 1)')
                        return False

                else:
                    if level3 in json_dict[level1][level2]:
                        try:
                            ref_config[level1][level2][level3] = bool(json_dict[level1][level2][level3])
                        except:
                            logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value should be a boolean (true, false, 0 or 1)')
                            return False

        return True

    def retrieve_config_number(json_dict, ref_config, level1, level2, level3=None):
        if level1 in json_dict:
            if level2 in json_dict[level1]:

                if level3 is None:
                    if is_number(json_dict[level1][level2]):
                        ref_config[level1][level2] = float(json_dict[level1][level2])
                    else:
                        logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a single number')
                        return False

                else:
                    if level3 in json_dict[level1][level2]:
                        if is_number(json_dict[level1][level2][level3]):
                            ref_config[level1][level2][level3] = float(json_dict[level1][level2][level3])
                        else:
                            logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value should be a single number')
                            return False

        return True

    def retrieve_config_range(json_dict, ref_config, level1, level2, level3=None):
        if level1 in json_dict:
            if level2 in json_dict[level1]:

                if level3 is None:
                    if is_valid_numeric_range(json_dict[level1][level2]):
                        ref_config[level1][level2] = (json_dict[level1][level2][0], json_dict[level1][level2][1])
                    else:
                        logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be an array of two numbers')
                        return False

                else:
                    if level3 in json_dict[level1][level2]:
                        if is_valid_numeric_range(json_dict[level1][level2][level3]):
                            ref_config[level1][level2][level3] = (json_dict[level1][level2][level3][0], json_dict[level1][level2][level3][1])
                        else:
                            logging.error(
                                'Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value should be an array of two numbers')
                            return False

        return True

    def retrieve_config_string(json_dict, ref_config, level1, level2, level3=None, options=None, case_sensitive=False):
        if level1 in json_dict:
            if level2 in json_dict[level1]:

                if level3 is None:

                    if isinstance(json_dict[level1][level2], str):
                        if options is None:
                            ref_config[level1][level2] = json_dict[level1][level2]
                        else:
                            value_cased = json_dict[level1][level2]
                            if not case_sensitive:
                                options = [option.lower() for option in options]
                                value_cased = value_cased.lower()
                            if value_cased in options:
                                ref_config[level1][level2] = json_dict[level1][level2]
                            else:
                                logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value can only be one of the following options: ' + str(options)[1:-1])
                                return False
                    else:
                        logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + ', the value should be a string')
                        return False

                else:
                    if level3 in json_dict[level1][level2]:
                        if isinstance(json_dict[level1][level2][level3], str):
                            if options is None:
                                ref_config[level1][level2][level3] = json_dict[level1][level2][level3]
                            else:
                                value_cased = json_dict[level1][level2][level3]
                                if not case_sensitive:
                                    options = [option.lower() for option in options]
                                    value_cased = value_cased.lower()
                                if value_cased in options:
                                    ref_config[level1][level2][level3] = json_dict[level1][level2][level3]
                                else:
                                    logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value can only be one of the following options: ' + str(options)[1:-1])
                                    return False
                        else:
                            logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value should be a string')
                            return False

        return True

    def retrieve_config_tuple(json_dict, ref_config, level1, level2, level3=None, options=None, case_sensitive=False):
        if level1 in json_dict:
            if level2 in json_dict[level1]:

                if level3 is None:
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

                else:
                    if level3 in json_dict[level1][level2]:
                        if isinstance(json_dict[level1][level2][level3], list):
                            if options is None:
                                ref_config[level1][level2][level3] = tuple(json_dict[level1][level2][level3])
                            else:
                                options_cased = options
                                values_cased = json_dict[level1][level2][level3]
                                if not case_sensitive:
                                    options_cased = [option.lower() for option in options]
                                    values_cased = [value.lower() for value in values_cased]
                                ref_config[level1][level2][level3] = list()
                                for value in values_cased:
                                    if value in options_cased:
                                        ref_config[level1][level2][level3].append(value)
                                    else:
                                        logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the following values are allowed: ' + str(options)[1:-1])
                                        return False
                                ref_config[level1][level2][level3] = tuple(ref_config[level1][level2][level3])
                        else:
                            logging.error('Invalid value in the configuration file for ' + level1 + '->' + level2 + '->' + level3 + ', the value should an array of strings')
                            return False

        return True

    #
    # retrieve the settings
    #

    # preprocessing settings
    if not retrieve_config_bool(json_config, config, 'preprocess', 'high_pass'):
        return False
    if not retrieve_config_bool(json_config, config, 'preprocess', 'early_re_referencing', 'enabled'):
        return False
    if not retrieve_config_string(json_config, config, 'preprocess', 'early_re_referencing', 'method', options=('CAR', 'CAR_headbox')):
        return False

    if not retrieve_config_string(json_config, config, 'preprocess', 'line_noise_removal', options=('off', 'json', '50', '60', '50hz', '60hz')):
        return False
    if config['preprocess']['line_noise_removal'].lower() == '50hz':
        config['preprocess']['line_noise_removal'] = '50'
    if config['preprocess']['line_noise_removal'].lower() == '60hz':
        config['preprocess']['line_noise_removal'] = '60'
    # TODO: load line noise removal, try also to accept a number instead of a string

    if not retrieve_config_bool(json_config, config, 'preprocess', 'late_re_referencing', 'enabled'):
        return False
    if not retrieve_config_string(json_config, config, 'preprocess', 'late_re_referencing', 'method', options=('CAR', 'CAR_headbox')):
        return False
    # TODO: load early re-referencing channels
    # TODO: load late re-referencing channels


    # trials settings
    if not retrieve_config_range(json_config, config, 'trials', 'trial_epoch'):
        return False
    if not retrieve_config_string(json_config, config, 'trials', 'out_of_bounds_handling', options=('error', 'first_last_only', 'allow')):
        return False
    config['trials']['out_of_bounds_handling'] = str(config['trials']['out_of_bounds_handling']).lower()
    if not retrieve_config_range(json_config, config, 'trials', 'baseline_epoch'):
        return False
    if not retrieve_config_string(json_config, config, 'trials', 'baseline_norm', options=('median', 'mean', 'none')):
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
    if not retrieve_config_tuple(json_config, config, 'channels', 'measured_types', options=VALID_CHANNEL_TYPES):
        return False
    if len(config['channels']['measured_types']) == 0:
        logging.error('Invalid value in the configuration file for channels->measured_types, at least one channel type should be given')
        return False
    config['channels']['measured_types'] = [value.upper() for value in config['channels']['measured_types']]
    if not retrieve_config_tuple(json_config, config, 'channels', 'stim_types', options=VALID_CHANNEL_TYPES):
        return False
    if len(config['channels']['stim_types']) == 0:
        logging.error('Invalid value in the configuration file for channels->stim_types, at least one channel type should be given')
        return False
    config['channels']['stim_types'] = [value.upper() for value in config['channels']['stim_types']]

    # cross-projection metric settings
    if not retrieve_config_bool(json_config, config, 'metrics', 'cross_proj', 'enabled'):
        return False
    if not retrieve_config_range(json_config, config, 'metrics', 'cross_proj', 'epoch'):
        return False

    # waveform metric settings
    if not retrieve_config_bool(json_config, config, 'metrics', 'waveform', 'enabled'):
        return False
    if not retrieve_config_range(json_config, config, 'metrics', 'waveform', 'epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'metrics', 'waveform', 'bandpass'):
        return False

    # evoked response detection settings
    if not retrieve_config_bool(json_config, config, 'detection', 'negative'):
        return False
    if not retrieve_config_bool(json_config, config, 'detection', 'positive'):
        return False
    if not retrieve_config_range(json_config, config, 'detection', 'peak_search_epoch'):
        return False
    if not retrieve_config_range(json_config, config, 'detection', 'response_search_epoch'):
        return False

    # detection methods
    if not retrieve_config_string(json_config, config, 'detection', 'method', options=('std_base', 'cross_proj', 'waveform')):
        return False
    # TODO: multiple options?

    config['detection'].pop('std_base', None)
    config['detection'].pop('cross_proj', None)
    config['detection'].pop('waveform', None)

    if config['detection']['method'] == 'std_base':
        config['detection']['std_base'] = dict()
        if not retrieve_config_range(json_config, config, 'detection', 'std_base', 'baseline_epoch'):
            return False
        if not retrieve_config_number(json_config, config, 'detection', 'std_base', 'baseline_threshold_factor'):
            return False
    elif config['detection']['method'] == 'cross_proj':
        config['detection']['cross_proj'] = dict()
        if not retrieve_config_number(json_config, config, 'detection', 'cross_proj', 'threshold'):
            return False
    elif config['detection']['method'] == 'waveform':
        config['detection']['waveform'] = dict()
        if not retrieve_config_number(json_config, config, 'detection', 'waveform', 'threshold'):
            return False

    # visualization settings
    if not retrieve_config_bool(json_config, config, 'visualization', 'negative'):
        return False
    if not retrieve_config_bool(json_config, config, 'visualization', 'positive'):
        return False
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
    config_str = '{\n' \
                 '    "preprocess": {\n' \
                 '        "high_pass":                        ' + ('true' if _config['preprocess']['high_pass'] else 'false') + ',\n' \
                 '        "early_re_referencing": {\n' \
                 '            "enabled":                      ' + ('true' if _config['preprocess']['early_re_referencing']['enabled'] else 'false') + ',\n' \
                 '            "method":                       "' + _config['preprocess']['early_re_referencing']['method'] + '",\n' \
                 '            "stim_excl_epoch":              [' + numbers_to_padded_string(_config['preprocess']['early_re_referencing']['stim_excl_epoch'], 16) + '],\n' \
                 '            "channel_types":                ' + json.dumps(_config['preprocess']['early_re_referencing']['channel_types']) + '\n' \
                 '        },\n' \
                 '        "line_noise_removal":               "' + _config['preprocess']['line_noise_removal'] + '",\n' \
                 '        "late_re_referencing": {\n' \
                 '            "enabled":                      ' + ('true' if _config['preprocess']['late_re_referencing']['enabled'] else 'false') + ',\n' \
                 '            "method":                       "' + _config['preprocess']['late_re_referencing']['method'] + '",\n' \
                 '            "stim_excl_epoch":              [' + numbers_to_padded_string(_config['preprocess']['late_re_referencing']['stim_excl_epoch'], 16) + '],\n' \
                 '            "channel_types":                ' + json.dumps(_config['preprocess']['late_re_referencing']['channel_types']) + '\n' \
                 '        }\n' \
                 '    },\n\n' \
                 '    "trials": {\n' \
                 '        "trial_epoch":                      [' + numbers_to_padded_string(_config['trials']['trial_epoch'], 16) + '],\n' \
                 '        "out_of_bounds_handling":           "' + _config['trials']['out_of_bounds_handling'] + '",\n' \
                 '        "baseline_epoch":                   [' + numbers_to_padded_string(_config['trials']['baseline_epoch'], 16) + '],\n' \
                 '        "baseline_norm":                    "' + _config['trials']['baseline_norm'] + '",\n' \
                 '        "concat_bidirectional_pairs":       ' + ('true' if _config['trials']['concat_bidirectional_pairs'] else 'false') + ',\n' \
                 '        "minimum_stimpair_trials":          ' + str(_config['trials']['minimum_stimpair_trials']) + '\n' \
                 '    },\n\n' \
                 '    "channels": {\n' \
                 '        "measured_types":                   ' + json.dumps(_config['channels']['measured_types']) + ',\n' \
                 '        "stim_types":                       ' + json.dumps(_config['channels']['stim_types']) + '\n' \
                 '    },\n\n' \
                 '    "metrics": {\n' \
                 '        "cross_proj": {\n' \
                 '            "enabled":                      ' + ('true' if _config['metrics']['cross_proj']['enabled'] else 'false') + ',\n' \
                 '            "epoch":                        [' + numbers_to_padded_string(_config['metrics']['cross_proj']['epoch'], 16) + ']\n' \
                 '        },\n' \
                 '        "waveform": {\n' \
                 '            "enabled":                      ' + ('true' if _config['metrics']['waveform']['enabled'] else 'false') + ',\n' \
                 '            "epoch":                        [' + numbers_to_padded_string(_config['metrics']['waveform']['epoch'], 16) + '],\n' \
                 '            "bandpass":                     [' + numbers_to_padded_string(_config['metrics']['waveform']['bandpass'], 16) + ']\n' \
                 '        }\n' \
                 '    },\n\n' \
                 '    "detection": {\n' \
                 '        "negative":                         ' + ('true' if _config['detection']['negative'] else 'false') + ',\n' \
                 '        "positive":                         ' + ('true' if _config['detection']['positive'] else 'false') + ',\n' \
                 '        "peak_search_epoch":                [' + numbers_to_padded_string(_config['detection']['peak_search_epoch'], 16) + '],\n' \
                 '        "response_search_epoch":            [' + numbers_to_padded_string(_config['detection']['response_search_epoch'], 16) + '],\n' \
                 '        "method":                           "' + _config['detection']['method'] + '",\n'

    if _config['detection']['method'] == 'std_base':
        config_str += '        "std_base": {\n' \
                      '            "baseline_epoch":               [' + numbers_to_padded_string(_config['detection']['std_base']['baseline_epoch'], 16) + '],\n' \
                      '            "baseline_threshold_factor":    ' + str(_config['detection']['std_base']['baseline_threshold_factor']) + '\n' \
                      '        }\n'
    elif _config['detection']['method'] == 'cross_proj':
        config_str += '        "cross_proj": {\n' \
                      '            "threshold":                    ' + str(_config['detection']['cross_proj']['threshold']) + '\n' \
                      '        }\n'
    elif _config['detection']['method'] == 'waveform':
        config_str += '        "waveform": {\n' \
                      '            "threshold":                    ' + str(_config['detection']['waveform']['threshold']) + '\n' \
                      '        }\n'

    config_str += '    },\n\n' \
                  '    "visualization": {\n' \
                  '        "negative":                         ' + ('true' if _config['visualization']['negative'] else 'false') + ',\n' \
                  '        "positive":                         ' + ('true' if _config['visualization']['positive'] else 'false') + ',\n' \
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
    Perform sanity checks on a given configuration

    Args:
        config (dict):                        The configuration to check

    Returns:
        bool:                                 True when passing all checks, False elsewise
    """

    def check_number_positive(ref_config, level1, level2, level3=None):

        if level3 is None:
            if ref_config[level1][level2] <= 0:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the input should be a positive value (> 0)')
                return False

        else:
            if ref_config[level1][level2][level3] < 0:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the input should be a positive value (> 0)')
                return False

        return True

    def check_epoch_start_after_onset(ref_config, level1, level2, level3=None):

        if level3 is None:
            if ref_config[level1][level2][0] < 0:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the epoch should start after the stimulus onset (>= 0s)')
                return False

        else:
            if ref_config[level1][level2][level3][0] < 0:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the epoch should start after the stimulus onset (>= 0s)')
                return False

        return True

    def check_epoch_within_trial(ref_config, level1, level2, level3=None):

        if level3 is None:
            if ref_config[level1][level2][0] < config['trials']['trial_epoch'][0]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given start-point (at ' + str(ref_config[level1][level2][0]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
                return False
            if ref_config[level1][level2][1] > config['trials']['trial_epoch'][1]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][1]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
                return False

        else:
            if ref_config[level1][level2][level3][0] < config['trials']['trial_epoch'][0]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the given start-point (at ' + str(ref_config[level1][level2][level3][0]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
                return False
            if ref_config[level1][level2][level3][1] > config['trials']['trial_epoch'][1]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][level3][1]) + 's) lies outside of the trial epoch (' + str(ref_config['trials']['trial_epoch'][0]) + 's - ' + str(ref_config['trials']['trial_epoch'][1]) + 's)')
                return False

        return True

    def check_range_order(ref_config, level1, level2, level3=None):

        if level3 is None:
            if ref_config[level1][level2][1] < ref_config[level1][level2][0]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][1]) + 's) lies before the start-point (at ' + str(ref_config[level1][level2][0]) + 's)')
                return False
            if ref_config[level1][level2][0] == ref_config[level1][level2][1]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'] parameter, the given start and end-point are the same (' + str(ref_config[level1][level2][0]) + 's)')
                return False

        else:
            if ref_config[level1][level2][level3][1] < ref_config[level1][level2][level3][0]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the given end-point (at ' + str(ref_config[level1][level2][level3][1]) + 's) lies before the start-point (at ' + str(ref_config[level1][level2][level3][0]) + 's)')
                return False
            if ref_config[level1][level2][level3][0] == ref_config[level1][level2][level3][1]:
                logging.error('Invalid [\'' + level1 + '\'][\'' + level2 + '\'][\'' + level3 + '\'] parameter, the given start and end-point are the same (' + str(ref_config[level1][level2][level3][0]) + 's)')
                return False

        return True

    # parameter start-end order
    if not check_range_order(config, 'trials', 'trial_epoch'):
        return False
    if not check_range_order(config, 'trials', 'baseline_epoch'):
        return False
    if not check_range_order(config, 'metrics', 'cross_proj', 'epoch'):
        return False
    if not check_range_order(config, 'metrics', 'waveform', 'epoch'):
        return False
    if not check_range_order(config, 'detection', 'peak_search_epoch'):
        return False
    if not check_range_order(config, 'detection', 'response_search_epoch'):
        return False
    if config['detection']['method'] == 'std_base':
        if not check_range_order(config, 'detection', 'std_base', 'baseline_epoch'):
            return False
    if not check_range_order(config, 'visualization', 'x_axis_epoch'):
        return False
    if not check_range_order(config, 'visualization', 'blank_stim_epoch'):
        return False

    # detection epoch parameters should be within trial epoch
    if not check_epoch_within_trial(config, 'metrics', 'cross_proj', 'epoch'):
        return False
    if not check_epoch_within_trial(config, 'metrics', 'waveform', 'epoch'):
        return False
    if not check_epoch_within_trial(config, 'detection', 'peak_search_epoch'):
        return False
    if not check_epoch_within_trial(config, 'detection', 'response_search_epoch'):
        return False
    if config['detection']['method'] == 'std_base':
        if not check_epoch_within_trial(config, 'detection', 'std_base', 'baseline_epoch'):
            return False

    if not check_epoch_within_trial(config, 'visualization', 'x_axis_epoch'):
        return False
    if not check_epoch_within_trial(config, 'visualization', 'blank_stim_epoch'):
        return False

    # trial epoch should start before the stimulus onset (routines in run rely on that)
    if config['trials']['trial_epoch'][0] >= 0:
        logging.error('Invalid [\'trials\'][\'trial_epoch\'] parameter, the epoch should start before the stimulus onset (< 0s)')
        return False

    # metric epochs should be after stimulus onset
    if not check_epoch_start_after_onset(config, 'metrics', 'cross_proj', 'epoch'):
        return False
    if not check_epoch_start_after_onset(config, 'metrics', 'waveform', 'epoch'):
        return False

    # detection peak search should be after stimulus onset
    if not check_epoch_start_after_onset(config, 'detection', 'peak_search_epoch'):
        return False
    if not check_epoch_start_after_onset(config, 'detection', 'response_search_epoch'):
        return False

    # the baseline threshold factor should be a positive number
    if config['detection']['method'] == 'std_base':
        if not check_number_positive(config, 'detection', 'std_base', 'baseline_threshold_factor'):
            return False
    elif config['detection']['method'] == 'cross_proj':
        if not check_number_positive(config, 'detection', 'cross_proj', 'threshold'):
            return False
    elif config['detection']['method'] == 'waveform':
        if not check_number_positive(config, 'detection', 'waveform', 'threshold'):
            return False

    # the waveform bandpass limits
    if config['metrics']['waveform']['bandpass'][0] <= 0:
        logging.error('Invalid [\'metrics\'][\'waveform\'][\'bandpass\'] parameter, the given lower cutoff frequency should be a positive number (' + str(config['metrics']['waveform']['bandpass'][0]) + ')')
        return False
    if config['metrics']['waveform']['bandpass'][1] <= 0:
        logging.error('Invalid [\'metrics\'][\'waveform\'][\'bandpass\'] parameter, the given upper cutoff frequency should be a positive number (' + str(config['metrics']['waveform']['bandpass'][1]) + ')')
        return False
    if config['metrics']['waveform']['bandpass'][1] < config['metrics']['waveform']['bandpass'][0]:
        logging.error('Invalid [\'metrics\'][\'waveform\'][\'bandpass\'] parameter, the upper cutoff frequency (' + str(config['metrics']['waveform']['bandpass'][1]) + ') is smaller than the lower cutoff frequency (' + str(config['metrics']['waveform']['bandpass'][0]) + ')')
        return False
    if config['metrics']['waveform']['bandpass'][0] == config['metrics']['waveform']['bandpass'][1]:
        logging.error('Invalid [\'metrics\'][\'waveform\'][\'bandpass\'] parameter, the given lower and upper cutoff frequencies are the same (' + str(config['metrics']['waveform']['bandpass'][0]) + ')')
        return False

    # return success
    return True


# initialize a variable with a default configuration dictionary for this module
_config = create_default_config()
