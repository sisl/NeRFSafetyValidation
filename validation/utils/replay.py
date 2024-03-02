import csv
import os
import numpy as np
from torch import norm
from tqdm import trange
from validation.simulators.BlenderSimulator import BlenderSimulator

def replay(start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn):
    '''
    This function reads a CSV file and for each row where the last column is 'True', 
    it creates a BlenderSimulator instance and runs it with a noise vector derived from columns 3-14 of the row.

    Parameters:
        csv_file (str): The path to the CSV file.
        start_state (torch.Tensor): The starting state of the simulator.
        end_state (torch.Tensor): The ending state of the simulator.
        agent_cfg (dict): The configuration for the agent.
        planner_cfg (dict): The configuration for the planner.
        camera_cfg (dict): The configuration for the camera.
        filter_cfg (dict): The configuration for the filter.
        get_rays_fn (function): A function to get rays.
        render_fn (function): A function to render the scene.
        blender_cfg (dict): The configuration for Blender.
        density_fn (function): A function to get the density of a point in space.
    '''

    file_list = os.listdir('results')
    csv_file_name = next((file for file in file_list if file.lower().endswith('.csv')), None)

    if csv_file_name:
        csv_file_path = os.path.join('results', csv_file_name)
        simulationNums = set()
        noise_vectors = []

        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[-1] == 'TRUE':
                    simulationNums.add(row[0])
                    noises = []
                    while True:
                        noise_vector = np.array(row[2:14], dtype=np.float32)
                        noises.append(noise_vector)
                        if row[-2] == 'TRUE':
                            break
                        row = next(reader, None)  
                    noise_vectors.append(noises)

            


    print(simulationNums)
    print(noise_vectors)
    simulator = BlenderSimulator(start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn)
    outputSimulationList = []
    everCollided = False
    simTrajLogLikelihood = 0

    print(f"Starting replay validation on BlenderSimulator")
    while simulationNums:
        simulator.reset()
        simulationNumber = simulationNums.pop()
        simulationSteps = noise_vectors.pop()
        # TODO change this from max steps to actual steps
        for step in trange(len(simulationSteps)):
            # pdb.set_trace()
            noise = simulationSteps[step]
            print(f"Step {step} with noise: {noise}")
            isCollision, collisionVal, currentPos = simulator.step(noise)
            outputStepList = [simulationNumber, step]

            # append the noises
            noiseList = noise.cpu().numpy()

            outputStepList.extend(noiseList)
            
            # append the sdf value and positions
            outputStepList.append(collisionVal)
            outputStepList.extend(currentPos)

            # find and append the trajectory likelihood, both for this step and the entire trajectory
            curLogLikelihood = trajectoryLikelihood(noiseList)
            outputStepList.append(curLogLikelihood)

            simTrajLogLikelihood += curLogLikelihood
            outputStepList.append(simTrajLogLikelihood)
            
            # output the collision value
            outputStepList.append(isCollision)
            
            # append the value of the step to the simulation data
            outputSimulationList.append(outputStepList)

            # if isCollision:
            #     collisions += 1
            #     stepsToCollision += stepNumber
            #     everCollided = True
            #     runBlenderOnFailure(self.blend_file, self.workspace, simulationNumber, stepNumber)
            #     break

        with open("./results/collisionValuesReplay.csv", "a") as csvFile:
            print(f"Noise List: {noiseList}")
            writer = csv.writer(csvFile)
            for outputStepList in outputSimulationList:
                outputStepList.append(everCollided)
                writer.writerow(outputStepList) 


def trajectoryLikelihood(self, noise):
    # get the likelihood of a noise measurement by finding each element's probability, logging each, and returning the sum
    likelihoods = norm.pdf(noise, loc = self.noise_mean_cpu, scale = self.noise_std_cpu)
    logLikelihoods = np.log(likelihoods)
    return logLikelihoods.sum()