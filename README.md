# N1Detect
A BIDS App for the automatic detection of early responses (N1) in CCEP data

## Usage
This App has the following command line arguments:

		usage: run.py [-h]
		              [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
		              bids_dir output_dir {participant,group}

		N1Detect BIDS App entry point script.

		positional arguments:
		  bids_dir              The directory with the input dataset formatted
		                        according to the BIDS standard.
		  output_dir            The directory where the output files should be stored.
		                        If you are running a group level analysis, this folder
		                        should be prepopulated with the results of
		                        the participant level analysis.
		  {participant,group}   Level of the analysis that will be performed. Multiple
		                        participant level analyses can be run independently
		                        (in parallel).

		optional arguments:
		  -h, --help            show this help message and exit
		  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
		                        The label(s) of the participant(s) that should be
		                        analyzed. The label corresponds to
		                        sub-<participant_label> from the BIDS spec (so it does
		                        not include "sub-"). If this parameter is not provided
		                        all subjects will be analyzed. Multiple participants
		                        can be specified with a space separated list.

To run it in participant level mode (for one participant):

    docker run -i --rm \
		-v /Users/max/data/ds001:/bids_dataset:ro \
		-v /Users/max/outputs:/outputs \
		bids/example \
		/bids_dataset /outputs participant --participant_label 01
