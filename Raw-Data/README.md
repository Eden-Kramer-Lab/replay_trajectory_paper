# Dataset description

This datasets contains neural recordings of ten male Long Evans rats (500-700 g, 4-9 months old) that were trained on a W-track spatial alternation task. 9 rats contributed to previous studies (Karlsson et al. 2008, 2009, Carr et al. 2012, Kay et al. 2016, 2020). Neural activity was recorded using tetrodes in CA1, CA2, CA3, MEC, Subiculum, and Dentate Gyrus depending on the rat.

This dataset has been processed to extract the LFP, sharp wave ripple times, clustered cells, multiunit spikes and associated spike features, and position of the animal using custom Frank lab Matlab code.

### Rats
1. Coriander
2. Bond
3. Chapati
4. Conley
5. Dave
6. Dudley
7. Egypt
8. Frank
9. Government
10. Remy

The data were collected by Dr. Mattias Karlsson (animals Bond, Frank, Conley, Dudley), Dr. Margaret Carr (animal Coriander), Dr. Kenneth Kay (animals Chapati, Dave), Drs. Kenneth Kay and Marielena Sosa (animal Egypt), Drs. Kenneth Kay and Jason E. Chung (animal Government), and Dr. Anna K. Gillespie (animal Remy).

Data from Drs. Margaret Carr and Mattias Karlsson was previously posted at CRCNS: http://crcns.org/data-sets/hc/hc-6. Any publications made using that data should cite the data set using the following:

```
Mattias Karlsson, Margaret Carr, Loren M. Frank (2015). Simultaneous extracellular recordings from hippocampal areas CA1 and CA3 (or MEC and CA1) from rats performing an alternation task in two W-shaped tracks that are geometrically identically but visually distinct.
CRCNS.org. http://dx.doi.org/10.6080/K0NK3BZJ
```

# Format of the data

### 1. Overview of directory structure

Each directory contains Matlab files corresponding to a single animal. Within each animal directory, there is a subfolder named “EEG”. This folder contains the local field potential recordings from each tetrode.

### 2. Naming conventions

All files are named based on the first three letters of the animal’s name, followed by the type of data they contain, following by a two digit number indicating the day of recording for that file. Thus, conspikes02.mat contains spiking data from Animal Conley for day 2 of data collection.

EEG files (which should be named “LFP” but are not for historical reasons) follow a slightly different naming convention, where each file consists of the first three letters of the animals name, (which we refer to as the animal short name) followed by the data type, followed by three numbers separated by dashes which give the day of recording, the epoch (defined below) and the tetrode whose data is contained in the file. Thus, coneeg02-3-05.mat contains data for day 2, epoch 3, tetrode 5.

### 3. Data organization

All datatypes are stored in nested cell arrays with the same organization.
  - The first level cell array indexes over days.
  - The second level cell array indexes over epochs, which are contiguous chunks of data where the animal is in one particular environment, such as on a track or in a
  rest box.
  - The third level cell array, if it exists, indexes over tetrodes.
  - The fourth level cell array, if it exists, indexes over clustered units (or cells if you
  prefer) within a tetrode.
  spikes{2}{3}{1}{4} would therefore contain data from day 2, epoch 3, tetrode 1 and unit 4.
  Note that because the data are generally stored by day, if you load conspikes02.mat, you will get a spike cell array where spikes{1} is empty, and spikes{2} has data.

### 3. Data files

##### {animal}cellinfo.mat #####
- **Description**: Defines basic measured characteristics of each neuron such as Spike Width and Mean Rate.
- **Format**:
  - 1 x {*n_days*} Matlab-cell
    - 1 x {*n_epochs*} Matlab-cell
      - 1 x {*n_tetrodes*} Matlab-cell
        - 1 x {n_cells} Matlab-cell
          - Cell Matlab-structure
            - `spikewidth`: duration of spike, in points sampled at 30 KHz, of spike from peak to trough
            - `meanrate`: mean rate of spikes (Hz)

##### {animal}spikes{day}.mat #####
- **Description**: Gives the spike times and other relevant information at spike times.
- **Format**:
  - 1 x {*n_days*} Matlab-cell
    - 1 x {*n_epochs*} Matlab-cell
      - 1 x {*n_tetrodes*} Matlab-cell
        - Spike Matlab-structure
          - fields: fields of data array
            - `time`: time of spike event in seconds
            - `x`: x-position of animal at spike
            - `y`: y-position of animal at spike
            - `dir`: head direction at spike
            - `amplitude`: amplitude of highest variance channel
            - `x-sm`: smoothed x-position of animal at spike
            - `y-sm`: smoothed y-position of animal at spike
            - `dir-sm`: smoothed head direction at spike

##### {animal}rawpos{day}.mat #####
- **Description**: Raw position data collected from the ccd camera.

##### **Name**: {*animal*}pos{*n_days*}.mat #####
- **Description**: Position of animal on track. Derived from the rawpos data structures. Timestamps are in seconds.
- **Format**:
  - 1 x {*n_days*}
      - 1 x {*n_epochs*}
          - position structure
              - arg: arguments used to derive the position structure from the rawpos information
              - descript: description of the data in the position structure
              - fields: labels for the data array
              - data: array with field labels
                  - `time`: time in session in seconds
                  - `x`: x-position of animal (cm)
                  - `y`: y-position of animal (cm)
                  - `dir`: head-direction of animal
                  - `vel`: velocity of animal (cm / s)
                  - `x-sm`: smoothed x-position of animal (cm)
                  - `y-sm`: smoothed y-position of animal (cm)
                  - `dir-sm`: smoothed head-direction of animal
                  - `vel-sm`: smoothed velocity of animal (cm / s)
              - `cmperpixel`: - centimeters per pixel, scales the ccd camera pixels to centimeters

##### {*animal*}linpos{*n_days*}.mat #####
- **Description**: Gives the animal's distance from a well (either at the starting position well or at the end of one of the arms). Derived from the pos data structure.

##### {*animal*}task{*n_days*}.mat #####
- **Description**: Defines the different task epochs for a single experimental session
- **Format**:
  - 1 x {*n_days*} Matlab-cell
      - 1 x {*n_epochs*} Matlab-cell:
          - task epoch Matlab-structure
              - `Type` - type of epoch (sleep, run, rest)
              - `Linear coord`: these are coordinates relative to user specified endpoints of track segments (only there if relevant to task epoch).
              - `Environment`: type of running track
                  -  `lin`: linear track
                  -  `wtr1`: w-track
                  -  `postsleep`: after sleeping
                  -  `presleep`: before sleeping

##### {*animal*}DIO{*n_days*}.mat #####
- **Description**: The *DIO* cell gives arrival/departure times at the end of each arm of the maze (as indicated by the IR motion sensors at the end of the wells) and the start/stop times for the output trigger to the reward pump. Timestamps are in 100 µsec units.
- **Format**:
  - 1 x {*n_days*}
      - 1 x {*n_epochs*}
          - 1 x {number of DIO-board pins}
              - non-empty cells correspond to active pins
              - there are two active pins for each reward well corresponding to either the IR motion sensor or the reward pump
              - DIO structure
                  - `pulsetimes`: start/stop time of activation
                  - `timesincelast`: time since last activation
                  - `pulselength`: duration of activation
                  - `pulseind`: index of pulse

##### {*animal*}ripples{*n_days*}-{epoch}-{tetrode}.mat #####
- **Description**: First processing step in the ripple identification process. Contains the ripples after the bandpass filtering on a single tetrode
- **Format**:

##### {*animal*}ripples{*n_days*}.mat #####
- **Description**: Intermediate processing step in the ripple identification process. Contains the ripples after the bandpass filtering, z-scoring, thresholding and segmenting the ripple on a single tetrode
- **Format**:
    - 1 x {*n_days*} Matlab-cell
        - 1 x {*n_epochs*} Matlab-cell
            - 1 x {*n_tetrodes*} Matlab-cell
                - `startind`: index of the start time of each ripple
                - `endind`: index of the end time of each ripple
                - `midind`: index of of the midpoint time of each ripple
                - `startime`: start time of the ripple
                - `endtime`: end time of the ripple
                - `midtime`: mid-point time of the ripple
                ...
##### {*animal*}candripples{*n_days*}.mat #####
- **Description**: Final processing step in the ripple identification process. Contains the ripples after combining tetrodes
- **Format**:
    - 1 x {*n_days*} Matlab-cell
        - 1 x {*n_epochs*} Matlab-cell

##### {*animal*}marks{*n_days*}.mat #####
- **Description**:
- **Format**:
  - 1 x {*n_days*} Matlab-cell
      - 1 x {*n_epochs*} Matlab-cell
          - 1 x {*n_tetrodes*} Matlab-cell
            - Marks Matlab structure
              - `data`, (n_time, n_mark_features)
                - columns correspond to spike amplitudes at time of spike for each tetrode wire

##### {*animal*}tetinfo.mat #####
- **Description**: Gives basic information about each tetrode such as the depth and number of neurons recorded, lists the valid electrodes
- **Format**:
    - 1 x {*n_days*} Matlab-cell
        - 1 x {*n_epochs*} Matlab-cell
            - 1 x {*n_tetrodes*} Matlab-cell
                - Tetrode Matlab-structure
                    - `depth`: depth of electrode, if specified, this is the number of 1/12ths of a turn of a 0x80 (80 threads / inch) screw from the bottom of the microdrive at which these data were collected. To convert to mm, multiply by 0.0265.
                    - `numcells`: number of neurons recorded on this tetrode
                    - `descrip`: Identifies the type of tetrode - riptet, {brain_area}REF
                    - `area`: brain area

##### {*animal*}eeg{*n_days*}-{*epoch*}-{*tetrode*}.mat #####
- **Description**: Gives the recorded local field potential for a tetrode.
- **Format**:
    - 1 x {*n_epochs*} Matlab-cell
        - 1 x {*n_tetrodes*} Matlab-cell
            - EEG Matlab-structure
                - `descript` - timestamps in hrs:min:sec
                - `fields` - fields of data array
                - `starttime` - time of first data sample (in seconds)
                - `samprate` - sampling rate in Hz
                - `data` - data array with columns corresponding to fields
                - `depth` - depth of electrode
##### {*animal*}eeggrnd{*n_days*}-{*epoch*}-{*tetrode*}.mat #####
- **Description**: Gives the LFP for a given ground tetrode (Ground wire located in the corpus collosum)
- **Format**:
    - 1 x {*n_epochs*} Matlab-cell
        - 1 x {*n_tetrodes*} Matlab-cell
            - EEG Matlab-structure
                - `descript` - timestamps in hrs:min:sec
                - `fields` - fields of data array
                - `starttime` - in seconds
                - `samprate` - sampling rate in Hz
                - `data` - data array
                - `depth` - depth of electrode
