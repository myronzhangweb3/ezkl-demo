# https://docs.ezkl.xyz/
# https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_all_public.ipynb#scrollTo=76f00d41

# here we create and (potentially train a model)

# make sure you have the dependencies required here already installed
from torch import nn
import ezkl
import os
import json
import torch

from install import install


# Defines the model
# we got convs, we got relu, we got linear layers
# What else could one want ????

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)

        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim=1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)

        return logits


async def main():
    circuit = MyModel()
    model_path = os.path.join('data/network.onnx')
    compiled_model_path = os.path.join('data/network.compiled')
    pk_path = os.path.join('data/test.pk')
    vk_path = os.path.join('data/test.vk')
    settings_path = os.path.join('data/settings.json')

    witness_path = os.path.join('data/witness.json')
    data_path = os.path.join('data/input.json')
    srs_path = os.path.join('data/kzg14.srs')

    shape = [1, 28, 28]
    # After training, export to onnx (network.onnx) and create a data file (input.json)
    x = 0.1 * torch.rand(1, *shape, requires_grad=True)

    # Flips the neural net into inference mode
    circuit.eval()

    # Export the model
    torch.onnx.export(circuit,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data=[data_array])

    # Serialize data into file:
    json.dump(data, open(data_path, 'w'))

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # "fixed" for params means that the committed to params are used for all proofs

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    assert res == True

    cal_path = os.path.join("data/calibration.json")

    data_array = (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data=[data_array])

    # Serialize data into file:
    json.dump(data, open(cal_path, 'w'))

    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = await ezkl.get_srs(settings_path, srs_path=srs_path)

    # now generate the witness file

    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK

    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # GENERATE A PROOF

    proof_path = os.path.join('data/test.pf')

    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path
    )

    print(res)
    assert os.path.isfile(proof_path)

    # VERIFY IT ON LOCAL

    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path
    )

    assert res == True
    print("verified")

    # VERIFY IT ON CHAIN
    verify_sol_code_path = os.path.join('data/verify.sol')
    verify_sol_abi_path = os.path.join('data/verify.abi')
    res = await ezkl.create_evm_verifier(
        vk_path,
        settings_path,
        verify_sol_code_path,
        verify_sol_abi_path,
        srs_path
    )
    assert res == True

    verify_contract_addr_file = "data/addr.txt"
    rpc_url = "http://127.0.0.1:3030"
    await ezkl.deploy_evm(
        addr_path=verify_contract_addr_file,
        rpc_url=rpc_url,
        sol_code_path=verify_sol_code_path
    )
    if os.path.exists(verify_contract_addr_file):
        with open(verify_contract_addr_file, 'r') as file:
            verify_contract_addr = file.read()
    else:
        print(f"File {verify_contract_addr_file} does not exist.")
        return
    res = await ezkl.verify_evm(
        addr_verifier=verify_contract_addr,
        proof_path=proof_path,
        rpc_url=rpc_url
    )
    assert res == True
    print("verified on chain")



import asyncio

asyncio.run(main())
