# N1Detect
A BIDS App for the automatic detection of early responses (N1) in CCEP data

This BIDS app was developed with support from the National Institute of Mental Health, R01MH122258 to DH.

## Usage

To launch an instance of the container and analyse some data in BIDS format, type:

```
$ docker run bids/N1Detect bids_dir output_dir [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```
For example, to run an analysis, type:

```
$ docker run -ti --rm \
  -v /path/to/local/bids/input/dataset/:/data \
  -v /path/to/local/output/:/output \
  bids/n1detect \
  /data /output --participant_label 01
```

----
To adjust the N1 detection and visualization settings, a JSON file can be passed using the ```--config_filepath [JSON_FILEPATH]``` parameter.
An example JSON of the standard settings has the following content:
```
{
    "trials": {
        "trial_epoch":                     [-1.0,        3.0],
        "baseline_epoch":                  [-1.0,       -0.1],
        "baseline_norm":                   "median",
        "concat_bidirectional_pairs":      true
    },

    "n1_detect": {
        "peak_search_epoch":               [ 0,          0.5],
        "n1_search_epoch":                 [ 0.009,     0.09],
        "n1_baseline_epoch":               [-1,         -0.1],
        "n1_baseline_threshold_factor":    3.4
    },

    "visualization": {
        "lim_epoch":                       [-0.2,          1],
        "stim_epoch":                      [-0.015,   0.0025],
        "generate_electrode_images":       false,
        "generate_stimpair_images":        false,
        "generate_matrix_images":          true
    }
}
```
For more information the settings...