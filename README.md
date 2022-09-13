# Evoked Response Detection
A python package and docker application for the automatic detection of evoked responses in CCEP data

## Usage

To launch an instance of the container and analyse data in BIDS format, type:

```
$ docker run multimodalneuro/n1detect bids_dir output_dir [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```
For example, to run an analysis, type:

```
$ docker run -ti --rm \
  -v /path/to/local/bids/input/dataset/:/data \
  -v /path/to/local/output/:/output \
  multimodalneuro/n1detect /data /output --participant_label 01 --skip_bids_validator
```

## Configure detection
To adjust the preprocessing, evoked response detection and visualization settings, a JSON file can be passed using the ```--config_filepath [JSON_FILEPATH]``` parameter.
An example JSON of the standard settings has the following content:
```
{
    "trials": {
        "trial_epoch":                      [-1.0,        3.0],
        "out_of_bounds_handling":           "first_last_only",
        "baseline_epoch":                   [-1.0,       -0.1],
        "baseline_norm":                    "median",
        "concat_bidirectional_pairs":       true,
        "minimum_stimpair_trials":          5
    },

    "channels": {
        "types":                            ["ECOG", "SEEG", "DBS"]
    },

    "detection": {
        "peak_search_epoch":                [ 0,          0.5],
        "response_search_epoch":            [ 0.009,     0.09],
        "method":                           "std_base",
        "std_base": {
            "baseline_epoch":               [-1,         -0.1],
            "baseline_threshold_factor":    3.4
        }
    },

    "visualization": {
        "x_axis_epoch":                     [-0.2,          1],
        "blank_stim_epoch":                 [-0.015,   0.0025],
        "generate_electrode_images":        false,
        "generate_stimpair_images":         false,
        "generate_matrix_images":           true
    }
}
```
For more information the settings...

## Acknowledgements

- Written by Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
- Local extremum detection method by Dorien van Blooijs & Dora Hermes (2018), with optimized parameters by Jaap van der Aar
- Dependencies:
  - PyMef by Jan Cimbalnik, Matt Stead, Ben Brinkmann, and Dan Crepeau (https://github.com/msel-source/pymef)
  - MNE-Python (https://mne.tools/)
  - BIDS-validator (https://github.com/bids-standard/bids-validator)
  - NumPy
  - SciPy
  - Pandas
  - KiwiSolver
  - Matplotlib
  - psutil

- This project was funded by the National Institute Of Mental Health of the National Institutes of Health Award Number R01MH122258 to Dora Hermes
