# C Code Generation for Neural Network Controllers

This project now supports automatic generation of C code from trained neural network models for quadrotor control.

## Overview

The C code generation feature converts trained neural network models (typically trained with Brax/JAX) into C code that can be deployed on embedded systems like the Crazyflie quadrotor.

## Requirements

- Python 3.11 (TensorFlow compatibility)
- TensorFlow 2.x
- JAX/Brax (for model loading)
- Other dependencies in `requirements.txt`

## Setup

1. **Create the conda environment:** (yml file coming soon)
   ```bash
   conda create -n quad_tf python=3.11 tensorflow -y
   conda activate quad_tf
   pip install -r requirements.txt
   ```

2. **Install additional dependencies:**
   ```bash
   conda activate quad_tf
   pip install brax jax jaxlib
   ```

## Usage

### Quick Start

Run the C code generation script:

```bash
conda activate quad_tf
python generate_c_code.py
```

### Manual Usage

You can also use the main script:

```bash
conda activate quad_tf
python main.py
```

### Programmatic Usage

```python
import quad_gen.get_models as get_models

# Generate C code from a model
get_models.save_result(
    model_dir="path/to/input_model",
    out_dir="path/to/output_model"
)
```

## Generated Files

The C code generation produces:

1. **`network_evaluate.c`** - The main C implementation containing:
   - Neural network structure definition
   - Weight matrices and bias vectors
   - Activation functions (linear, sigmoid, relu)
   - `networkEvaluate()` function for forward pass
   - Output assignment to control structure

2. **`params.pkl`** - Backup of original model parameters

## Generated C Code Structure

The generated C code includes:

- **Header includes**: `#include "network_evaluate.h"`
- **Activation functions**: `linear()`, `sigmoid()`, `relu()`
- **Network structure**: Static arrays defining layer dimensions
- **Weight matrices**: All neural network weights as 2D arrays
- **Bias vectors**: All bias terms as 1D arrays
- **Output arrays**: Static arrays for intermediate layer outputs
- **Main function**: `networkEvaluate()` that performs the forward pass

## Model Compatibility

The system supports:
- **JAX/Flax models**: Trained with Brax or similar frameworks
- **TensorFlow models**: Traditional TensorFlow models
- **Parameter structures**: Handles various parameter formats

## Integration with Crazyflie

The generated C code is designed to integrate with the Crazyflie firmware:

1. Include the generated `network_evaluate.c` in your firmware
2. Call `networkEvaluate()` from your controller
3. The function expects:
   - `control_n`: Control structure for thrust outputs
   - `state_array`: Input state vector (17-dimensional)

## Example Integration

```c
// In your controller
float state_array[17];  // Your state vector
struct control_t_n control_n;

// Fill state_array with current state
// ...

// Run neural network
networkEvaluate(&control_n, state_array);

// Use outputs
float thrust_0 = control_n.thrust_0;
float thrust_1 = control_n.thrust_1;
float thrust_2 = control_n.thrust_2;
float thrust_3 = control_n.thrust_3;
```

## Troubleshooting

### Common Issues

1. **TensorFlow import errors**: Make sure you're using Python 3.11 and the correct conda environment
2. **Model loading errors**: Ensure the model was trained with compatible frameworks
3. **Parameter structure errors**: The system automatically detects and handles different parameter formats

### Debugging

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify the model file exists and is readable
3. Ensure the output directory is writable
4. Check the console output for specific error messages

## Technical Details

### Neural Network Architecture

The generated C code supports:
- **Input layer**: 17 dimensions (state vector)
- **Hidden layers**: Configurable (typically 32 neurons each)
- **Output layer**: 4 dimensions (thrust commands)
- **Activation**: tanh for hidden layers, linear for output

### Performance Considerations

- **Memory usage**: All weights are stored as static arrays
- **Computation**: Simple matrix-vector operations
- **Optimization**: Suitable for real-time embedded systems

## Contributing

To extend the C code generation:

1. **Add new activation functions**: Modify `code_blocks.py`
2. **Support new model formats**: Update `gaussian_mlp.py`
3. **Modify output format**: Adjust the C code generation logic

## License

This C code generation feature is part of the quad_sim2multireal project. 