# Evoked Response Detection
A python package and docker application for the automatic detection of evoked responses in SPES/CCEP data

## Docker Usage

To launch an instance of the container and analyse data in BIDS format, in the command-line interface/terminal:

```
docker run multimodalneuro/erdetect <bids_dir>:/data <output_dir>:/output [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```
For example, to run an analysis, type:

```
docker run -ti --rm \
-v /local_bids_data_root/:/data \
-v /local_output_path/:/output \
multimodalneuro/erdetect /data /output --participant_label 01
```


## Python Usage

1. First install ERdetect, in the command-line run:
```
pip install erdetect
```

2. To run:
- a) From the commandline:
```
python -m erdetect ~/bids_data ~/output/ [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```

- b) To process a subset directly in a python script:
```
import erdetect
erdetect.process_subset('/bids_data_root/subj-01/ieeg/sub-01_run-06.edf', '/output_path/')
```



## Configure detection
From the command-line, a JSON file can be passed using the ```--config_filepath [JSON_FILEPATH]``` parameter to adjust the preprocessing, the evoked response detection and the visualization settings.
An example JSON containing the standard settings looks as follows:
```
{
    "preprocess": {
        "high_pass":                        false,
        "line_noise_removal":               "off",
        "early_re_referencing": {
            "enabled":                      false,
            "method":                       "CAR",
            "stim_excl_epoch":              [-1.0,        2.0]
        }
    },
	
    "trials": {
        "trial_epoch":                      [-1.0,        2.0],
        "out_of_bounds_handling":           "first_last_only",
        "baseline_epoch":                   [-0.5,      -0.02],
        "baseline_norm":                    "median",
        "concat_bidirectional_pairs":       true,
        "minimum_stimpair_trials":          5
    },

    "channels": {
        "measured_types":                   ["ECOG", "SEEG", "DBS"],
        "stim_types":                       ["ECOG", "SEEG", "DBS"]
    },

    "detection": {
        "negative":                         true,
        "positive":                         false,
        "peak_search_epoch":                [ 0,          0.5],
        "response_search_epoch":            [ 0.009,     0.09],
        "method":                           "std_base",
        "std_base": {
            "baseline_epoch":               [-1,         -0.1],
            "baseline_threshold_factor":    3.4
        }
    },

    "visualization": {
        "negative":                         true,
        "positive":                         false,
        "x_axis_epoch":                     [-0.2,          1],
        "blank_stim_epoch":                 [-0.015,   0.0025],
        "generate_electrode_images":        true,
        "generate_stimpair_images":         true,
        "generate_matrix_images":           true
    }
}
```


## Acknowledgements

- Written by Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
- Local extremum detection method by Dorien van Blooijs & Dora Hermes (2018), with optimized parameters by Jaap van der Aar
- Dependencies:
  - IeegPrep (https://github.com/MultimodalNeuroimagingLab/ieegprep)
  - BIDS-validator (https://github.com/bids-standard/bids-validator)
  - NumPy
  - SciPy
  - Matplotlib

- This project was funded by the National Institute Of Mental Health of the National Institutes of Health Award Number R01MH122258 to Dora Hermes
