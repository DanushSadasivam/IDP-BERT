# IDP-BERT
Property Prediction for Intrinsically Disordered Proteins (IDPs) using Language Model. ([Paper](https://arxiv.org/abs/2403.19762))

<br>

## Getting Started
* Clone this repository
* `cd IDP-BERT`
* Install the required packages (`pip install -r requirements.txt`)
* Create train and test splits by running `python data/split_data.py`

<br>

## How to Run Inference
* Update the `config.yaml` file with the desired parameters
* Run `python main.py` to train the model
* Place the inference dataset in a numpy array named `X` saved as `inference_data.npz` in the directory `./data/`.
* Update file name of the trained model in the `run_name` field in `config.yaml`.
* Run `python inference.py` to obtain predictions.
* The predictions file can be found in `./data/inference_results/`.
