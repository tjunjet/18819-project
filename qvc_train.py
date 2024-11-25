import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, StatePreparation, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from datetime import datetime

from tqdm import tqdm


# Initialize the optimizer
optimizer = COBYLA(maxiter=100)

# Initialize the Sampler primitive
sampler = StatevectorSampler() #backend=qpu_backend)

# Defining a callback to see the iteration
def verbose_callback(weights, objective_value):
    """
    Verbose callback function to monitor the training process during fit.

    Args:
        weights (np.ndarray): Current variational circuit parameters.
        objective_value (float): Current loss value.
    """
    print(f"Iteration: Objective Value = {objective_value:.4f}, Weights = {weights}")

# Initialize a global progress bar
progress_bar = tqdm(total=100, desc="Training Progress")  # Replace 100 with your optimizer's max iterations

def tqdm_callback(weights, objective_value):
    """
    Callback function to update the tqdm progress bar during training.

    Args:
        weights (np.ndarray): Current variational circuit parameters.
        objective_value (float): Current value of the loss function.
    """
    progress_bar.update(1)
    progress_bar.set_postfix({"Loss": objective_value})


def get_train_test_data(opts):
    # Data loading
    ###########################################################
    X_train = np.loadtxt('middle_data/X_train.txt', delimiter=' ')
    X_test  = np.loadtxt('middle_data/X_test.txt', delimiter=' ')

    Y_train = np.loadtxt('middle_data/Y_train.txt', delimiter=' ')
    Y_test  = np.loadtxt('middle_data/Y_test.txt', delimiter=' ')

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # PCA for feature extraction
    ###########################################################
    # Dimensionality reduction
    pca = PCA(n_components=opts.num_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Normalize data to the range [norm min, norm max] (use the training set to fit the scaler)
    scaler = MinMaxScaler(feature_range=(opts.norm_min, opts.norm_max))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure labels are binary (if not already binary)
    Y_train = (Y_train > np.median(Y_train)).astype(int)
    Y_test = (Y_test > np.median(Y_test)).astype(int)

    print(f"X_train shape after selection: {X_train.shape}")
    print(f"X_test shape after selection: {X_test.shape}")

    return X_train, Y_train, X_test, Y_test

def get_feature_map(opts):
    if opts.fm_circuit == 'ZZFeatureMap':
        feature_map = ZZFeatureMap(feature_dimension=opts.num_features, 
                                reps=opts.fm_reps, 
                                entanglement=opts.fm_entanglement)
    elif opts.fm_circuit == 'PauliFeatureMap':
        feature_map = PauliFeatureMap(feature_dimension=opts.num_features, 
                                reps=opts.fm_reps, 
                                entanglement=opts.fm_entanglement)
    elif opts.fm_circuit == 'StatePreparation':
        feature_map = StatePreparation(feature_dimension=opts.num_features, 
                                reps=opts.fm_reps, 
                                entanglement=opts.fm_entanglement)
    else:
        raise Exception("Unsupported feature map circuit.")
        
    return feature_map

def main(opts):
    """
    Main training function
    """
    X_train, Y_train, X_test, Y_test = get_train_test_data(opts)

    # Feature map
    feature_map = get_feature_map(opts)
    
    # Ansatz circuit
    ansatz = RealAmplitudes(num_qubits=opts.num_features, 
                            reps=opts.ansatz_reps, 
                            entanglement=opts.ansatz_entanglement)

    
    # Build the Quantum Variational Classifier
    # Num Qubits is specified as None, so that we just use the number of qubits
    # specified by the quantum feature map
    vqc = VQC(feature_map=feature_map, 
            ansatz=ansatz, 
            optimizer=optimizer,
            loss="cross_entropy", # default
            sampler=sampler,
            callback=tqdm_callback
            )
    
    pretrain_datetime = datetime.now()
    vqc.fit(X_train, Y_train)

    # Testing the VQC
    train_score_q4 = vqc.score(X_train, Y_train)
    test_score_q4 = vqc.score(X_test, Y_test)
    current_datetime = datetime.now()

    time_taken = current_datetime.minute - pretrain_datetime.minute

    print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
    print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

    print(f"Time taken for training:     {time_taken:.2f}")

    # Write to output file
    formatted_datetime = current_datetime.strftime('%Y-%m-%d%H:%M:%S')
    output_file = os.path.join(opts.output_dir, f"{formatted_datetime}.txt")
    with open(output_file, "w") as file:
        file.write("Chosen Parameters:\n")
        file.write(f"  num_features: {opts.num_features}\n")
        file.write(f"  norm_min: {opts.norm_min}\n")
        file.write(f"  norm_max: {opts.norm_max}\n")
        file.write(f"  fm_circuit: {opts.fm_circuit}\n")
        file.write(f"  fm_reps: {opts.fm_reps}\n")
        file.write(f"  fm_entanglement: {opts.fm_entanglement}\n")
        file.write(f"  ansatz_reps: {opts.ansatz_reps}\n")
        file.write(f"  ansatz_entanglement: {opts.ansatz_entanglement}\n")
        file.write("===================================================\n")

        file.write(f"Quantum VQC on the training dataset: {train_score_q4:.2f}\n")
        file.write(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}\n")
        file.write(f"Time taken:                          {time_taken:.2f}\n")

    print(f"Parameters written to {output_file}")

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Data Preprocessing
    parser.add_argument('--num_features', type=int, default=2)
    parser.add_argument('--norm_min', type=int, default=-1)
    parser.add_argument('--norm_max', type=int, default=1)
    
    # Data Encoding (feature map)
    parser.add_argument('--fm_circuit', type=str, default='ZZFeatureMap')
    parser.add_argument('--fm_reps', type=int, default=1)
    parser.add_argument('--fm_entanglement', type=str, default='linear')

    # Ansatz Circuit
    parser.add_argument('--ansatz_reps', type=int, default=1)
    parser.add_argument('--ansatz_entanglement', type=str, default='linear') # can try 'full', 'circular', 'reverse_linear'

    # Output Dir
    parser.add_argument('--output_dir', type=str, default='results') 

    return parser

def print_opts_params(opts):
    print("Chosen Parameters:")
    print(f"  num_features: {opts.num_features}")
    print(f"  norm_min: {opts.norm_min}")
    print(f"  norm_max: {opts.norm_max}")
    print(f"  fm_circuit: {opts.fm_circuit}")
    print(f"  fm_reps: {opts.fm_reps}")
    print(f"  fm_entanglement: {opts.fm_entanglement}")
    print(f"  ansatz_reps: {opts.ansatz_reps}")
    print(f"  ansatz_entanglement: {opts.ansatz_entanglement}")
    print("===================================================")


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    print_opts_params(opts)
    main(opts)
