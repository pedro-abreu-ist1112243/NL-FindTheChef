# Natural Language Processing Project - 2025

This repository has the code for the Natural Language Processing course at Instituto Superior Técnico, 2025.

The object of the project is to predict the chef who wrote a recipe, given a number of features.

# Running the Project

By default, the train script skips grid search to save time. To enable grid search, add the `--grid-search` flag when running the training script.
You can train the model with the files in `data/` by running:

```bash
uv run src/train.py
# or with grid search:
uv run src/train.py --grid-search
```

You can run predictions on the unlabeled test set by running:

```bash
uv run src/predict.py
```

You can train or predict with custom data through the `main.py` script:
```bash
uv run src/main.py
Usage: python src/main.py <operation> [<args>...]

Operations:
    train <training_data_path> <artifacts_output_path>
        Train a model using the provided training data and save artifacts to the specified path.
        This operation may take a long time due to BERT embeddings and grid searching.
    predict <artifacts_path> <recipes_path> <predictions_output_path> <grid_search (optional) (default: False)>
        Predict the chef for the given recipes using the trained model.
        Results are saved to the specified output path.
        Grid search can be enabled by setting the last argument to True.
    help
        Show this help message.
```

# Note on CUDA

This project uses PyTorch and can leverage CUDA for GPU acceleration if available. You might need to change the `tool.uv.sources` key in `pyproject.toml` according to the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

The fine-tuned BERT model is, unfortunately, much too large to submit to GitHub (~1.2GB), so you will need to train it yourself.
