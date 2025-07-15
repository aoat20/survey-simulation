from survey_simulation import SEASSimulation

# To run scenario 4 in manual mode
ss = SEASSimulation(scenario_n=2,
                    mode='manual')

# To set up scenario 4 in test mode
ss = SEASSimulation(scenario_n=2,
                    mode='test',
                    plotter=True)

# Run until mission is finished
while not ss.mission_finished:
    # Advance to the next time step
    ss.next_step()

    # Get the observations
    obs = ss.get_obs()

    # Do stuff based on the observations
    if obs['time_s'] == 300:
        ss.set_waypoints([[420_000, 5_560_000],
                          [425_000, 5_560_000]])
    if obs['time_s'] == 2000:
        ss.set_course(30)
    if obs['time_s'] == 2400:
        ss.set_speed(30)
    if obs['time_s'] > 3500 and ss.course_reached:
        ss.set_course(-60)

print(ss.termination_reason)

# Playback one of the log files
SEASSimulation(mode='playback',
               scenario_n='logs/log_1.json',
               )
