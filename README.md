# Pointer Network for NoC Optimization

This repository contains the implementation of a **Pointer Network** designed for optimizing **Network-on-Chip (NoC) routing and scheduling**. The network is trained using supervised learning, reinforcement learning (RL), and tested for performance in various configurations of multi-core systems.

The implementation leverages PyTorch and supports CUDA for GPU acceleration. It allows training, reinforcement learning, and testing using different numbers of cores and configurations.

---

## Prerequisites

#### Installing Dependencies

To install the required Python packages, use the provided `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```

#### Optional: Installing PyTorch with CUDA

If you want to use GPU acceleration, make sure to install a CUDA-compatible version of PyTorch. Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and use the appropriate command for your system. For example:

```bash
# Example: Installing PyTorch with CUDA 11.3 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

---

## Usage

Connect to the DGX server by the following command:

    ssh -i <rsa_id_filename> <username>:<ip_addr>

    bash

Ensure that the CONDA environment is activated on the DGX server by running the following command on terminal:

    source /opt/apps/anaconda3/bin/activate

Run the script using the `main()` function, which supports three modes: `train`, `rl`, and `test`.


### Arguments

| Argument           | Description                                      | Default Value |
|--------------------|--------------------------------------------------|---------------|
| `--mode`           | Mode of operation: `train`, `rl`, or `test`.     | `test`        |
| `--num_samples`    | Number of samples during decoding.               | `1`           |
| `--batch_size`     | Batch size for training or RL.                   | `64`          |
| `--lr`             | Learning rate.                                   | `1e-8`        |
| `--wt_decay`       | Weight decay for optimizer regularization.       | `1e-5`        |
| `--factor`         | Scheduler factor for learning rate adjustments.  | `0.001`       |
| `--patience`       | Early stopping patience.                         | `100`         |
| `--dropout`        | Dropout probability in the network.              | `0.1`         |
| `--n_layers`       | Number of layers in the encoder/decoder.         | `2`           |
| `--input_dim`      | Input dimension for the encoder.                 | `2`           |
| `--hidden_dim`     | Hidden dimension size for the network.           | `2`           |
| `--epoch`          | Maximum number of epochs for training or RL.     | `10000`       |
| `--num_cores`      | Number of cores in the NoC system.               | `7`           |
| `--num_test_iter`  | Number of test iterations to be run              | `50`          |
| `--alpha`          | Discount factor for RL.                          | `0.99`        |

---

### Examples

1. **Training**:
   ```bash
   python main.py --mode train --num_cores 32 --batch_size 64 --lr 1e-4
   ```

2. **Reinforcement Learning (RL)**:
   ```bash
   python main.py --mode rl --num_cores 14 --alpha 0.9 --epoch 5000
   ```

3. **Testing**:
   ```bash
   python main.py --mode test --num_cores 32
   ```

---

## Detailed Workflow

### 1. Training
- **Objective**: Train the Pointer Network using supervised learning to optimize mappings of I/O pairs to cores.
- **Process**:
  1. Initialize the PointerNet model and optimizer.
  2. Load data for the given number of cores.
  3. Train the model for the specified number of epochs.
  4. Save the trained model and optimizer states for future use.

### 2. Reinforcement Learning (RL)
- **Objective**: Fine-tune the Pointer Network using RL to improve performance further.
- **Process**:
  1. Load the pre-trained PointerNet model.
  2. Use RL-based techniques (e.g., policy gradients) to improve the model's decision-making.
  3. Save the fine-tuned model and optimizer states.

### 3. Testing
- **Objective**: Evaluate the trained model on a test dataset to measure performance.
- **Process**:
  1. Load the trained PointerNet model.
  2. Prepare the test dataset.
  3. Run the model for multiple test iterations (e.g., 1000) and print the outputs of the test iterations.
  4. Copy paste these into `ga.py` file located at `/home/kanchanm/Desktop/noc_ga/` and run ga.py

---

## Datasets

The datasets for training are named as `data_<num_cores>cores.txt` and for testing are named as `bm_<num_cores>cores.txt`. They will be in the `/home/kanchanm/Desktop/noc/` folder. 

## Results and Outputs

1. **Model States**:
   - Trained model weights are saved in `.pt` files (e.g., `ordered_active_search_32cores.pt`).

2. **Optimizer States**:
   - Optimizer states are saved for resuming training (e.g., `ordered_opt_32cores.pt`).

3. **Performance Metrics**:
   - Best mapping and corresponding cost are printed during testing.

4. **Logging**:
   - Outputs such as `max(costs)` and `min(costs)` help evaluate model performance on the test dataset.

---