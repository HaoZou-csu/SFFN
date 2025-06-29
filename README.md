# Stepwise feature fusion network for predicting the formation energy of inorganic crystals

This project is a deep learning framework designed to predict the physicochemical properties of materials using advanced graph neural networks and Transformer models. The core of this framework is the `SFFN` model, which combines the strengths of both **electronic**, **atomic** and **attention** models. It leverages a multi-modal learning approach to extract rich features from a material's chemical composition for high-accuracy property regression.

The project is highly configurable; all training parameters, including model architecture, data loading, optimizers, and the training pipeline, are managed through JSON files.

## Core Features

  * **Hybrid Model Architecture**: Fuses the graph-based **STA** model with the Transformer-based **SFA** model to capture both structural and compositional information of materials.
  * **Flexible Data Loading**: The `CombinedDataLoader` supports loading and combining various feature sets from multiple sources, such as Roost, CrabNet, Magpie, Meredig, and more.
  * **K-Fold Cross-Validation**: Built-in support for K-fold cross-validation allows for robust model evaluation.
  * **Highly Configurable**: Easily manage all experiment parameters like model architecture, learning rate, batch size, and loss functions through a JSON configuration file.
  * **Powerful Trainer Selection**: The `train.py` script automatically selects the appropriate trainer based on the configuration. For the `Combine` model type used in this project, it selects the `MMTrainer` (Multi-Modal Trainer).
  * **Standardized Training Pipeline**: Supports learning rate scheduling (`StepLR`), early stopping, resuming from checkpoints, and TensorBoard visualization.

## Project Structure (Inferred)

The project structure is as follows:

```
.
├── config/
│   └── config_combine_reg_fold0.json   # Experiment configuration file
├── data/
│   ├── MP/
│   │   ├── mp_roost_hf.csv             # Data required for Roost model
│   │   └── mp_crabnet_hf.csv           # Data required for CrabNet model
├── dataset/
│   ├── magpie_fea.npy                  # Magpie features
│   └── ...                             # Other feature files
├── model/
│   └── __init__.py                     # Contains model definitions like Roost_CrabNet_reg
├── data_loader/
│   └── __init__.py                     # Contains data loaders like CombinedDataLoader
├── trainer/
│   └── __init__.py                     # Contains trainers like MMTrainer, GraghTrainer
├── loss.py                             # Defines loss functions (e.g., L1_loss)
├── metric.py                           # Defines evaluation metrics (e.g., rmse, mae)
├── parse_config.py                     # Configuration file parser
├── train.py                            # Main training script
└── utils.py                            # Utility functions
```

## Environment Setup

1.  Clone this repository.

2.  Install the required Python dependencies. It is recommended to use a virtual environment.

    ```bash
    # This script automates the creation of a new Conda environment and installs all necessary dependencies.
    
    # -- Step 1: Create a new Conda environment named 'feature_comb' with Python 3.10 --
    # -n specifies the environment name
    # -y automatically confirms all prompts
    conda create -n feature_comb python=3.10 -y
    
    # -- Step 2: Install PyTorch and its related libraries in the new environment --
    # Use 'conda run' to execute a command within the specified environment
    # Specify versions for PyTorch, torchvision, torchaudio, and explicitly use CUDA 11.8
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # -- Step 3: Install PyTorch Geometric --
    # PyG is a graph neural network library and should be installed after PyTorch
    pip install torch_geometric
    
    # -- Step 4: Install materials science and model interpretation libraries --
    # Install pymatgen, matminer, jarvis-tools, and shap from the conda-forge channel in one go
    conda install pymatgen matminer jarvis-tools shap -c conda-forge -y
    
    # -- Environment setup complete --
    ```

    *Note: Please install a version of PyTorch that is compatible with your CUDA toolkit.*

## Configuration File Explained

The JSON configuration file is the core of the project, defining all training details. Below is an analysis of the key sections from `config_combine_reg_fold0.json`:

  * **`name`**: "Roost\_Crab\_Li\_S"

      * An identifier for the experiment, used for logging and saving models.

  * **`arch`**:

      * Defines the model architecture. In this case, it is `Roost_CrabNet_reg`, a regression model combining Roost and CrabNet. Parameters like `elem_emb_len`, `n_graph`, and `d_model` define the network's internal dimensions and layers.

  * **`data_loader`**:

      * The type is `CombinedDataLoader`, indicating it loads and merges multiple data sources.
      * Fields like `roost_data_dir` and `crabnet_data_dir` specify paths to different feature data files.
      * `batch_size`: 128.
      * `use_Kfold`: `true`, enabling K-fold cross-validation. This config is for the first fold (`nth_fold: 0`) out of 5 total folds (`total_fold: 5`).

  * **`optimizer`**:

      * Uses the `AdamW` optimizer with a learning rate of `1e-4`.

  * **`loss`**:

      * Uses `L1_loss` (Mean Absolute Error) as the loss function.

  * **`metrics`**:

      * Evaluation metrics are `rmse` (Root Mean Squared Error) and `mae` (Mean Absolute Error).

  * **`lr_scheduler`**:

      * Uses the `StepLR` learning rate scheduler, which multiplies the learning rate by `0.1` every `50` epochs.

  * **`trainer`**:

      * `epochs`: Train for `100` epochs.
      * `monitor`: "min val\_loss", which tracks the validation loss for early stopping and model saving.
      * `early_stop`: `200`, stopping the training if validation loss does not improve for 200 epochs.
      * `save_dir`: The directory where model checkpoints are saved.

  * **`model_type`**: "Combine"

      * This is a critical field that determines which trainer to use in `train.py`. When the value is `Combine`, the script instantiates the `MMTrainer` (Multi-Modal Trainer) to execute the training process.

## How to Run

Launch training using the `train.py` script. You must specify a configuration file.

1.  **Start Training**:
    Use the `-c` or `--config` argument to specify the path to your configuration file.

    ```bash
    python train.py --config config/config_combine_reg_fold0.json
    ```

2.  **Resume Training**:
    If training was interrupted, you can resume from the last saved checkpoint using the `-r` or `--resume` argument.

    ```bash
    python train.py --resume saved/Roost_Crab_Li_S/checkpoint-epoch10.pth
    ```

3.  **Specify GPU**:
    Use the `-d` or `--device` argument to specify which GPU to use.

    ```bash
    python train.py --config <your_config.json> --device 0
    ```

4.  **Override Configuration from Command Line**:
    The script supports modifying configuration parameters directly from the command line for quick experiments.

      * Change learning rate:
        ```bash
        python train.py --config <your_config.json> --lr 0.0005
        ```
      * Change batch size:
        ```bash
        python train.py --config <your_config.json> --bs 256
        ```
