import subprocess
# take in a state vector and output the grid coordinate
# taken from nav/quad_plot.py
def stateToGridCoord(state):
    grid_size = 100//5 # side//kernel_size
    state_float = grid_size*(state[:3] + 1) / 2
    state_coord = tuple(int(state_float[i]) for i in range(3))
    return state_coord

def runBlenderOnFailure(blend_file, workspace, step):
    bevel_depth = 0.02      # Size of the curve visualized in blender
    subprocess.run(['blender', blend_file, '-P', 'validation/utils/viz_failures_blend.py', '--background', '--', workspace, str(bevel_depth), str(step)])

# You can call the function like this:
# run_blender_on_failure(blend_file, opt.workspace, bevel_depth)
