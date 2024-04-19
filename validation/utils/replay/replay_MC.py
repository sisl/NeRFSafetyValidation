import csv
import os
import numpy as np
from scipy.stats import norm
import torch
from tqdm import trange
from validation.simulators.BlenderSimulator import BlenderSimulator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from validation.utils.blenderUtils import runBlenderOnFailure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def replay_MC(start_state, end_state, noise_mean, noise_std, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn, blend_file, workspace, seed):
    '''
    This function reads a CSV file and for each row where the last column is 'True', 
    it creates a BlenderSimulator instance and runs it with a noise vector derived from columns 3-14 of the row.

    Parameters:
        csv_file (str): The path to the CSV file.
        start_state (torch.Tensor): The starting state of the simulator.
        end_state (torch.Tensor): The ending state of the simulator.
        noise_mean (torch.Tensor): Means of disturbances.
        noise_std (torch.Tensor): Standard deviation of noises to use.
        agent_cfg (dict): The configuration for the agent.
        planner_cfg (dict): The configuration for the planner.
        camera_cfg (dict): The configuration for the camera.
        filter_cfg (dict): The configuration for the filter.
        get_rays_fn (function): A function to get rays.
        render_fn (function): A function to render the scene.
        blender_cfg (dict): The configuration for Blender.
        density_fn (function): A function to get the density of a point in space.
    '''

    # read from csv resulting from simulations
    file_list = os.listdir('results')
    csv_file_name = next((file for file in file_list if file.lower().endswith('.csv')), None)

    if csv_file_name:
        csv_file_path = os.path.join('results', csv_file_name)
        simulationData = {}
        simulationResult = {}

        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                simulationNumber = row[0]
                noise_vector = torch.from_numpy(np.array(row[2:14], dtype=np.float32)).to(device)
                if simulationNumber not in simulationData:
                    simulationData[simulationNumber] = []
                    simulationResult[simulationNumber] = []
                simulationData[simulationNumber].append(noise_vector)
                simulationResult[simulationNumber].append([row[-2], row[-1]])

    # clear existing csv
    if os.path.exists("results/replays/collisionValuesReplay.csv"):
        os.remove("results/replays/collisionValuesReplay.csv")

    # counts
    tp_count_step, tn_count_step, fp_count_step, fn_count_step = 0, 0, 0, 0
    tp_count_traj, tn_count_traj, fp_count_traj, fn_count_traj = 0, 0, 0, 0

    # run replay validation
    simulator = BlenderSimulator(start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn, seed)
    print(f"Starting replay validation on BlenderSimulator")
    for simulationNumber, simulationSteps in simulationData.items():
        simulator.reset()
        outputSimulationList = []
        simTrajLogLikelihood = 0
        everCollided = False
        print(f"Replaying simulation {simulationNumber} with {len(simulationSteps)} steps!")
        for step in trange(len(simulationSteps)):
            noise = simulationSteps[step]
            print(f"Replaying step {step} with noise: {noise}")
            isCollision, collisionVal, currentPos = simulator.step(noise)
            outputStepList = [simulationNumber, step]

            # append the noises
            noiseList = noise.cpu().numpy()

            outputStepList.extend(noiseList)
            
            # append the sdf value and positions
            outputStepList.append(collisionVal)
            outputStepList.extend(currentPos)

            # find and append the trajectory likelihood, both for this step and the entire trajectory
            curLogLikelihood = trajectoryLikelihood(noiseList, noise_mean, noise_std)
            outputStepList.append(curLogLikelihood)

            simTrajLogLikelihood += curLogLikelihood
            outputStepList.append(simTrajLogLikelihood)
            
            # output the collision value
            outputStepList.append(isCollision)
            
            # append the value of the step to the simulation data
            outputSimulationList.append(outputStepList)

            # count by step
            nerf_condition = True if simulationResult[simulationNumber][step][0].upper() == "TRUE" else False
            tp_count_step += isCollision and nerf_condition
            fn_count_step += isCollision and not nerf_condition
            fp_count_step += not isCollision and nerf_condition
            tn_count_step += not isCollision and not nerf_condition

            if isCollision:
                everCollided = True
                # count the remaining steps after collision as false negatives
                remaining_steps = len(simulationSteps) - step - 1
                runBlenderOnFailure(blend_file, workspace, simulationNumber, step)
                fn_count_step += remaining_steps
                break
        if not everCollided:
            # visualize simulation at the end if no collision occurred
            runBlenderOnFailure(blend_file, workspace, simulationNumber, len(simulationSteps)-1)

        # count by simulation
        nerf_traj_condition = True if simulationResult[simulationNumber][-1][1].upper() == "TRUE" else False
        tp_count_traj += everCollided and nerf_traj_condition
        fn_count_traj += everCollided and not nerf_traj_condition
        fp_count_traj += not everCollided and nerf_traj_condition
        tn_count_traj += not everCollided and not nerf_traj_condition

        os.makedirs('results/replays', exist_ok=True)
        with open("results/replays/collisionValuesReplay.csv", "a") as csvFile:
            writer = csv.writer(csvFile)
            for outputStepList in outputSimulationList:
                outputStepList.append(everCollided)
                writer.writerow(outputStepList) 

    createConfusionMatrix(tp_count_step, tn_count_step, fp_count_step, fn_count_step, "step")
    createConfusionMatrix(tp_count_traj, tn_count_traj, fp_count_traj, fn_count_traj, "traj")


def trajectoryLikelihood(noise, noise_mean_cpu, noise_std_cpu):
    # get the likelihood of a noise measurement by finding each element's probability, logging each, and returning the sum
    likelihoods = norm.pdf(noise, loc = noise_mean_cpu.cpu().numpy(), scale = noise_std_cpu.cpu().numpy())
    logLikelihoods = np.log(likelihoods)
    return logLikelihoods.sum()

def createConfusionMatrix(tp, tn, fp, fn, name):
    plt.close('all')
    # load data into numpy array
    conf_matrix = np.array([[tn, fn], [fp, tp]])
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=['False', 'True'], index=['False', 'True'])

    # display confusion matrix using seaborn
    sns.heatmap(conf_matrix_df, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Blender Simulator Collision')
    plt.ylabel('NeRF Simulator Collision')
    plt.title(f'Confusion Matrix ({name})')
    plt.savefig(f'results/confusion_matrix_{name}.png')
    plt.show()