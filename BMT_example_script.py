from survey_simulation import SEASSimulation

# Set up scenario 4 in test mode
ss = SEASSimulation(scenario_n=4,
                    mode='test')

# Run until
while not ss.mission_finished:
    ss.next_step()
    obs = ss.get_obs()

    # Do stuff based on the observations
    if obs['time_s'] == 200:
        ss.set_course(30)
    if obs['time_s'] == 240:
        ss.set_speed(30)
    if obs['time_s'] > 350 and ss.course_reached:
        ss.set_course(-60)

print(ss.termination_reason)
