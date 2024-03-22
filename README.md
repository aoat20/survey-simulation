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
Import the package and initialise a sim object with mode, param filepath and save directory:
```python 
import survey_simulation
ss = survey_simulation.SurveySimulation(mode,
                                       params_filepath,
                                       save_dir)

```

`mode` can be "manual", "test" or "playback". `params_filepath` is the location of the parameter file location. `save_dir` is the directory in which to save log files. Log files will be saved to individual folders in that folder with incrementing episode numbers.

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
"move" action outputs four data objects:
```python
t, cov_map, contacts, occ_grid = ss.new_action('move', [x,y])
```
- "t" is the remaining mission time
- "cov_map" is the coverage from the pass, with the value being the scan direction.
- "contacts" is a  list of detections with fields:
  - `det_n` - detection index
  - `x` - x coordinate
  - `y` - y coordinate
  - `range` - range at which scan was performed (currently unused)
  - `angle` - angle from which the detection was made
  - `scan_n` - index of the scan in which the detection was made
  - `group_n` - index of the group to which the detection belongs
- "occ_grid" is the occupancy grid in image pixel coordinates. 1 is an occupied location and 0 is unoccupied and can therefore be traveled to. 

To save the episode logs:
```python
ss.save_episode()
```

### Playback
Playback mode is for playing back episode logs. "left" and "right" keys go backwards and forwards through actions.
`save_dir` is directory of your logs and the extra argument `ep_n` is the episode number to load.

## Demo

```python
import survey_simulation
import numpy as np

# example manual run
ss = survey_simulation.SurveySimulation('manual',
                                       save_dir='data',
                                       params_filepath='params.txt')

# example test run
ss = survey_simulation.SurveySimulation('test',
                                       save_dir='data',
                                       params_filepath='params.txt')
for n in range(100):
    rnd_mv = np.random.randint(0,100,size=(2)).tolist()
    t, cov_map, contacts, occ_grid = ss.new_action('move', rnd_mv)
    
    # At two arbitrary steps, demo group and ungroup actions
    if n==45: 
        ss.new_action('group', [0,1,2])
    if n==65:
        ss.new_action('ungroup', [0])
        ss.new_action('group', [1,3,4])
# Save the episode log
ss.save_episode()

# Save to a specific episode number (CAUTION. This will delete the episode if it already exists)
episode_number = 2
ss.save_episode(episode_number)

# Playing back data
ss_pb = survey_simulation.SurveySimulation('playback',
                                          save_dir='data',
                                          ep_n=2)

```
## Parameter file
The parameter file needs to be a txt file with the following entries:
```
# Interface settings
rt: 0
play_speed: 1

# Agent properties
agent_start: (55., 190.) 
scan_width: 20.
leadinleadout: 5.
min_scan_l: 10.
nadir_width: 3.
agent_speed: 5.

# Map properties
#map_n: 1
#map_path: "maps/Map1.png"
scan_area_lims: (0, 200, 0, 120)
map_area_lims: (0, 200, 0, 120) 
grid_res: 10

# Target and clutter stats
n_targets: 3 
det_probs: (0.4, 0.9) 
loc_uncertainty: 3.
n_clutter: 8 
det_prob_clutter: 0.1 

# Mission parameters
time_lim: 500.
min_scan_angle_diff: 30 
N_looks: 6 
N_angles: 6
#rand_seed: 50
```

Notes:
- The hash key can be used for comments. 
- `map_n` can be any value 1-3, each for a different default map environment. If `agent_start` is omitted will default to a preset shore side location.
- `map_path` is used for custom maps and should point to the directory of a png with transparency indicating the water. Remember to set `agent_start` parameter. 
- If both `map_n` and `map_path` are omitted, a blank map will be used and the `scan_area_lims` will be used to generate the size of the map. `scan_area_lims` and `map_area_lims` are ignored if using a map.
- All of these parameters can be modified by passing the relevant argument to the class upon instantiation of the SurveySimulation object, for example, to change the agent start position:
```python 
ss = survey_simulation.SurveySimulation('manual',
                                       agent_start=[20,50]))
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
    4 update 25 27
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
rt: 0
play_speed: 1
agent_start: [0, 0]
scan_width: 20.0
leadinleadout: 5.0
min_scan_l: 10.0
nadir_width: 3.0
agent_speed: 5.0
scan_area_lims: [0, 200, 0, 120]
map_area_lims: [0, 200, 0, 120]
grid_res: 10
n_targets: 3
det_probs: [0.4, 0.9]
loc_uncertainty: 3.0
n_clutter: 8
det_prob_clutter: 0.1
time_lim: 500.0
min_scan_angle_diff: 30
N_looks: 6
N_angles: 6
rand_seed: 50
```
