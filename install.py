# check if notebook is in colab
import os


def install():
    try:
        import ezkl
        import onnx
    except:
        # install ezkl
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ezkl==11.2.2"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx==1.16.1"])

        if not os.path.exists("data"):
            os.makedirs("data")


install()
