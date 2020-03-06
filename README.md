# N1Detect
A BIDS App for the automatic detection of early responses (N1) in CCEP data

## Usage

To launch an instance of the container and analyse some data in BIDS format, type:

```
$ docker run bids/N1Detect bids_dir output_dir level [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```
For example, to run an analysis in ```participant``` level mode, type:

```
$ docker run -ti --rm \
  -v /path/to/local/bids/input/dataset/:/data \
  -v /path/to/local/output/:/output \
  bids/N1Detect \
  /data /output participant --participant_label 01
```

For example, to run an analysis in ```group``` level mode with a user-defined pipeline, type:

```
$ docker run -ti --rm \
  -v /path/to/local/bids/input/dataset/:/data \
  -v /path/to/local/output/:/output \
  bids/N1Detect \
  /data /output group
```


