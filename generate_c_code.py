#!/usr/bin/env python3
"""
Simple script to generate C code from trained neural network models.
This script enables C code generation for quadrotor control.
"""

import quad_gen.get_models as get_models


def main():
    print("=== Neural Network to C Code Generator ===")
    print("Starting C code generation...")
    
    get_models.save_result(
        model_dir="/Users/devesh/Desktop/Coding/research/college/sim2multireal/quad_sim2multireal/input_model",
        out_dir="/Users/devesh/Desktop/Coding/research/college/sim2multireal/quad_sim2multireal/output_model",
    )
    
    print("C code generation completed successfully!")
    print("Generated files:")
    print("- network_evaluate.c: Neural network implementation in C")
    print("- params.pkl: Original model parameters (backup)")


if __name__ == "__main__":
    main() 