# N1Detect
A BIDS App for the automatic detection of early responses (N1) in CCEP data

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
To adjust the N1 detection and visualization settings, a JSON file can be passed using the ```--config [JSON_FILEPATH]``` parameter.
An example JSON of the standard settings has the following content:
```
{
    "signal_extract": {
        "trial_epoch_range":    [-1,       3.0],
        "baseline_range":       [-1,      -0.1],
    },
    
    "n1_detect": {
        "peak_search_range":    [ 0,       0.5],
        "n1_search_range":      [ 0.009,   0.09],
        "baseline_std_range":   [-1,      -0.1],
        "baseline_std_fact":    3.4,
    },

    "visualization": {
        "x_axis_range":         [-0.2,    1],
        "x_stim_range":         [-0.015,  0.0025],
    }    
}
```
For more information the settings...