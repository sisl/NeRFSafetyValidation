import torch
import argparse
from validation.simulators.NerfSimulator import NerfSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################### MAIN LOOP ##########################################
def validate(simulator):
    # TODO
    return

####################### END OF MAIN LOOP ##########################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    opt = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    simulator = NerfSimulator()
  
    # Main loop
    validate(simulator)
    
    end_text = 'End of validation'
    print(f'{end_text:.^20}')
