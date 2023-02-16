# Pet Adoption Speed Prediction

## Introduction

In this project, we aim to develop algorithms to predict the adoptability of pets, based on the pet's listing on PetFinder. The goal is to predict the speed at which a pet is adopted, which is determined by the speed at which all of the pets are adopted if the profile represents a group of pets. The data includes text, tabular, and image data.

The successful algorithms will be adapted into AI tools that guide shelters and rescuers around the world on improving their pet profiles' appeal. This will help reduce animal suffering and euthanization.

## Codebase

**`notebooks`:** **Explore and visualize the data** 

**`tasks`** : **Convenience scripts for running frequent tests and training commands**

**`training`**: **Logic for the training itself**

- **`model_core`: the core code of were the model lives (p.e. `cat_recognizer`, `text_classifier`, `tumor detector`, etc)**
    - **`datasets`**: **Logic for downloading, preprocessing, augmenting, and loading data**
    - **`models`: Models wrap networks and add functionality like loss functions. saving, loading, and training**
    - **`networks` : Code for constructing neural networks (dumb input | output mappings)**
    - **`tests`: Regression tests for the models code. Make sure a trained model performs well on important examples.**
    - **`weights` : Weights of the production model**
    - `predictor.py`: **wrapper for model that allows you to do inference**
    - `utils.py`

**`api`**: **Web server serving predictions. DockerFiles, Unit Tests, Flask,  etc.** 

**`evaluation`**: **Run the validation tests** 

**`experiment_manager`**: **Settings of your experiment manager (**p.e. wandb, tensorboard**)**

**`data`**: **use it for data versioning, storing data examples and metadata of your datasets. During training use it to store your raw and processed data but don't push or save the datasets into the repo.** 

