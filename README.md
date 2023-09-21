# survey-simulation

Python package for simulating a survey agent

## Installation and updating
For virtual environment: 
```bash 
python -m venv venv
source venv/bin/activate
```

Install using pip and rerun to update:

```bash
pip install git+https://github.com/aoat20/survey-simulation
```

## Usage
Import the package and initialise a sim object with mode, param file location and save location:
```python 
import surveyimulation
ss = surveysimulation.SurveySimulation(mode,
                                       params_loc,
                                       save_loc)

```

`mode` can be "manual", "test" or "playback". `params_loc` is the location of the parameter file location. `save_loc` is the folder in which to save log files. Log files will be saved to individual folders with incrementing episode numbers.

### Manual 
Manual mode is for user operation to generate training data.

Click to go to next coordinate OR manually type next coordinate, followed by enter.

- "z" key to switch between snap mode and free mode.
- "shift" key to switch between move mode and group mode. In group mode, click on desired contacts, then press "a" key to group. To ungroup, while in group mode, click on the group.
- "#" key will reveal the target locations and end the episode. After ending the episode, press "=" to save the episode logs.

### Test 
Test mode is for interacting with the simulator in Python. After instantiating a simulator object, use new_action function to "move", "group" and "ungroup".

```python 
ss.new_action('move',[x, y])
ss.new_action('group', [c0, c1, c2])
ss.new_action('ungroup',[g0])
```
To save the episode logs:
```python
ss.save_episode()
```

### Playback
Playback mode is for playing back episode logs. "left" and "right" keys go backwards and forwards through actions.

## Demo
Demo mode is for playing back data logs. 

```python
import surveysimulation
import numpy as np

# example random run
ss = surveysimulation.SurveySimulation('test',
                                       save_loc='data')
for n in range(100):
    rnd_mv = np.random.randint(0,100,size=(2)).tolist()
    t, cov_map, contacts = ss.new_action('move', rnd_mv)
    
    # At two arbitrary steps, demo group and ungroup actions
    if n==45: 
        ss.new_action('group', [0,1,2])
    if n==65:
        ss.new_action('ungroup', [0])
        ss.new_action('group', [1,3,4])
# Save the episode log
ss.save_episode()

# Playing back data
ss_pb = surveysimulation.SurveySimulation('playback',
                                          save_loc='data/Episode0')

```

## Log Files Format
For an episode with an ID number:

### Actions
Contains "move", "group" and "ungroup" actions for each action id:
move to position (x, y):
```
    actionID move x y 
```
group contacts c0, c1, c2 to form group g0:
```
    actionID group g0 c0 c1 c2
```
ungroup group g1:
```
    actionID ungroup g1
```
eg. 
Episode2_ACTIONS.txt
```
    0 move 60 50
    1 move 20 30
    2 group 0 0 2
    3 ungroup 0
    4 move 35 20
    5 move 10 15
    ...
```

### Observations
Contains observations for the corresponding action ID in the format:
```
    actionID time_remaining contactID contactX contactY contactRange contactAngle
```
or if there's no contacts:
```
  actionID time_remaining
```
eg.
EpisodeID_OBSERVATIONS.txt
 ```
    0 100
    1 88 0 30 20 5 45
    1 88 1 5 12 8 225
    2 60
    3 55 
    4 45
    5 38 2 7 10 3 30
    ...
```

### Coverage maps
Map for each scan stored in a folder "COVERAGE" on each move containing a matrix of nans with the areas covered shown as numbers indicating the direction from which they were scanned.

eg. for episode 2 action 5 which has a 6x4 map area
COVERAGE/Episode2_action5_COVERAGE.mat
```
    nan nan nan 90 90 nan 
    nan nan 90 90 nan nan
    nan nan nan nan nan nan
    nan nan nan nan nan nan 
```

### Ground truth
Showing the position of each target and each clutter, as well as the corresponding orientation in the format: 
```
    objectID class x y th 
```
class can be either "target" or "clutter", with the latter not having any orientation
eg. 
Episode2_TRUTH.txt
 ```
   0 target 10 20 45
    1 target 30 40 60
    2 clutter 1 6
    3 clutter 50 45
```

### Meta data
Stored in the EpisodeID_META.txt, contains the following parameters which are required to recreate the episode. eg. 
```
agent_start: [0.0, 0.0]
scan_width: 20.0
leadinleadout: 5.0
min_scan_l: 10.0
nadir_width: 3.0
agent_speed: 20.0
scan_area_lims: [0, 200, 0, 100]
map_area_lims: [-20, 220, -20, 120]
n_targets: 2
n_clutter: 5
det_probs: [0.4, 0.9]
loc_uncertainty: 3.0
det_prob_clutter: 0.1
time_lim: 500.0
min_scan_angle_diff: 30
N_looks: 6
N_angles: 6
grid_res: 10
rand_seed: 100
```
