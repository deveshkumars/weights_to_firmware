#!/usr/bin/env python3
"""
Simple script to generate C code from trained neural network models.
This script enables C code generation for quadrotor control.
"""

import quad_gen.get_models as get_models
import os


def main():

    print("=== Neural Network to C Code Generator ===")
    print("Starting C code generation...")
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Running from directory: {current_dir}")

    get_models.save_result(
        model_dir=current_dir + "/input_model",
        out_dir=current_dir + "/output_model",
        osi=False,
        absolute_path=True
    )
    
    print("C code generation completed successfully!")
    print("Generated files:")
    print("- network_evaluate.c: Neural network implementation in C")
    print("- params.pkl: Original model parameters (backup)")


if __name__ == "__main__":
    main() 