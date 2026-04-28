# tinyGPT

This is an implementation of a tiny GPT.

## Repo layout

```text
tinyGPT/
|-- config/
|   |-- model_config.py
|   `-- train_config.py
|-- data/
|   `-- tiny_shakespeare.txt
|-- scripts/
|   |-- prepare_data.py
|   |-- train.py
|   `-- generate.py
|-- tinygpt/
|   |-- data.py
|   |-- generation.py
|   |-- model.py
|   |-- training.py
|   `-- utils.py
|-- homeworkGPT.ipynb
`-- requirements.txt
```

## Setup

Create and activate a virtual environment, then install the dependencies.

Use Python 3.11 or 3.12 for the cleanest PyTorch install path. The current workspace interpreter is Python 3.14, which is ahead of typical PyTorch wheel support.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Prepare the dataset

Convert the raw text corpus into train/validation tensors plus tokenizer metadata.

```powershell
python scripts/prepare_data.py --input data/tiny_shakespeare.txt --out-dir data/processed
```

This creates:

- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/meta.json`

## 2. Pretrain the model

Train the default model from the notebook configuration and save a checkpoint plus training history.

```powershell
python scripts/train.py --prepared-dir data/processed --device auto
```

Useful overrides:

```powershell
python scripts/train.py --prepared-dir data/processed --context-length 256 --n-layers 6 --n-heads 6 --n-embd 384 --batch-size 24 --max-iters 2000 --eval-interval 200 --learning-rate 3e-4 --device cuda
```

Training artifacts are written to:

- `artifacts/tinygpt.pt`
- `artifacts/train_history.json`

## 3. Run generation

Load the trained checkpoint and sample text from a prompt.

```powershell
python scripts/generate.py --prepared-dir data/processed --checkpoint-path artifacts/tinygpt.pt --prompt "ROMEO:"
```

Example with more control over sampling:

```powershell
python scripts/generate.py --prepared-dir data/processed --checkpoint-path artifacts/tinygpt.pt --prompt "To be, or not to be" --max-new-tokens 300 --temperature 0.8
```

## 4. Develop further

The repo is split so you can iterate by concern:

- `config/` contains the model and training hyperparameters.
- `tinygpt/model.py` contains the transformer implementation.
- `tinygpt/data.py` handles corpus loading, encoding, and saved metadata.
- `tinygpt/training.py` contains batching, loss estimation, training, and checkpointing.
- `tinygpt/generation.py` contains autoregressive sampling.

## Relationship to the notebook

The notebook is left in place as the original working reference. The reusable logic now lives in the Python package and CLI scripts so future experiments do not need to stay trapped inside a single static notebook.
