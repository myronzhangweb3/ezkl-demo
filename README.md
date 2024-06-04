# EZKL Test

This is a test example of [ezkl](https://docs.ezkl.xyz/). The test model is a simple convolutional neural network containing two convolutional layers and two fully connected layers, using ReLU as the activation function. It can be used to process batches of data with input shapes of (32, 1, 28, 28) and ultimately outputs predicted scores for 10 categories corresponding to each sample.
Proofs are generated and then tested for validation on the chain

## Dependencies

- Python>=3.7
- ezkl=11.2.2
- onnx=1.16.1

## Usage

### Run a local block chain

```shell
# install anvil if you haven't already
cargo install --git https://github.com/foundry-rs/foundry --profile local --locked anvil

# spin up a local EVM through anvil in a separate terminal 
anvil -p 3030
```

### Install

#### EZKL

Install as [ezkl installing document](https://docs.ezkl.xyz/installing)

#### Python Environment

```shell
python3 install.py
```

### Run

```shell
python3 main.py
```