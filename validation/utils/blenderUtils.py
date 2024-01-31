# take in a state vector and output the grid coordinate
# taken from nav/quad_plot.py
def stateToGridCoord(state):
    grid_size = 100//5 # side//kernel_size
    state_float = grid_size*(state[:3] + 1) / 2
    state_coord = tuple(int(state_float[i]) for i in range(3))
    return state_coord