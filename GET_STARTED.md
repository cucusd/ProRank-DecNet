## Environment
Ensure that Python ≥ 3.8 is installed. This project is developed with PyTorch. Please follow the official PyTorch instructionsto install PyTorch(https://pytorch.org/) and and [torchvision](https://pytorch.org/vision/stable/index.html). Then, install the Hugging Face Transformers library：
```
pip install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


## Training

Our training instructions are as follows:

```
python train_SOTA_mix.py
```

## Evaluation
The evaluation instructions are as follows:
```
python test_acc_1.py
```
