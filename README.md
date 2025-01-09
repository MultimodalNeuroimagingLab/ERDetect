# Evoked Response Detection
A python package and docker application for the automatic detection of evoked responses in SPES/CCEP data

## Python Usage

1. First install ERdetect, in the command-line run:
```
pip install erdetect
```

2. To run:
- a) With a graphical user interface:
```
python -m erdetect ~/bids_data ~/output/ --gui
```

- b) From the commandline:
```
python -m erdetect ~/bids_data ~/output/ [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
```

- c) To process a subset directly in a python script:
```
import erdetect
erdetect.process_subset('/bids_data_root/subj-01/ieeg/sub-01_run-06.edf', '/output_path/')
```

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



## Configuration & Documentation

General documentation can be found [here](https://github.com/MultimodalNeuroimagingLab/erdetect/wiki/).

The tool can be configured by three means:
- Graphical User Interface (GUI)
- Command-line, arguments and options can be found [here](https://github.com/MultimodalNeuroimagingLab/erdetect/wiki/Configuration#command-line-arguments)
- JSON input configuration file, usage documentation can be found [here](https://github.com/MultimodalNeuroimagingLab/erdetect/wiki/Configuration#json-input-configuration-file)


## Acknowledgements

- Written by Max van den Boom (Multimodal Neuroimaging Lab, Mayo Clinic, Rochester MN)
- Deviation from baseline method by Dorien van Blooijs & Dora Hermes (2018), with optimized parameters by Jaap van der Aar
- Adapted the peak-finder algorithm by Nathanael Yoder, see [Matlab version](http://www.mathworks.com/matlabcentral/fileexchange/25500).
- Dependencies:
  - IeegPrep (https://github.com/MultimodalNeuroimagingLab/ieegprep)
  - BIDS-validator (https://github.com/bids-standard/bids-validator)
  - NumPy
  - SciPy
  - Matplotlib

- This project was funded by the National Institute Of Mental Health of the National Institutes of Health Award Number R01MH122258 to Dora Hermes
