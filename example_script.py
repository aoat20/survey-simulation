from survey_simulation import SurveySimulationGrid
import numpy as np

# Running manual mode
ss = SurveySimulationGrid('manual')

# Running test mode
ss = SurveySimulationGrid('test',
                    save_dir='data')

for n in range(500):
    rnd_heading = np.random.randint(-70,70)

    ss.new_action('move', rnd_heading)
    obs = ss.next_step()

    if ss.end_episode:
        ss.reset()

# Running playback
ss = SurveySimulationGrid('playback', 
                            save_dir='data/',
                            ep_n=2)