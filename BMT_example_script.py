from survey_simulation import SEASSimulation

# To run scenario 4 in manual mode
ss = SEASSimulation(scenario_n=4,
                    mode='manual')

# To set up scenario 4 in test mode
ss = SEASSimulation(scenario_n=4,
                    mode='test',
                    plotter=True)

# Run until mission is finished
while not ss.mission_finished:
    # Advance to the next time step
    ss.next_step()

    # Get the observations
    obs = ss.get_obs()

    # Do stuff based on the observations
    if obs['time_s'] == 200:
        ss.set_course(30)
    if obs['time_s'] == 240:
        ss.set_speed(30)
    if obs['time_s'] > 350 and ss.course_reached:
        ss.set_course(-60)


print(ss.termination_reason)
