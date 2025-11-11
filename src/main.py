# Defer imports to functions to reduce initial load time
import sys


def _print_help(program_name: str):
    usage = f"""\
Usage: python {program_name} <operation> [<args>...]

Operations:
    train <training_data_path> <artifacts_output_path>
        Train a model using the provided training data and save artifacts to the specified path.
        This operation may take a very long time due to BERT embeddings and grid searching.
    predict <artifacts_path> <bert_model_path> <recipes_path> <predictions_output_path>
        Predict the chef for the given recipes using the trained model.
        Results are saved to the specified output path.
    help
        Show this help message.
    """

    print(usage)


def _train(args):
    if len(args) < 2:
        print(
            "Usage: python main.py train <training_data_path> <artifacts_output_path> <grid_search (optional, default False)>"
        )
        return

    import train

    train.main(
        args[0], args[1], grid_search=len(args) > 2 and args[2].lower() == "true"
    )


def _predict(args):
    if len(args) != 4:
        print(
            "Usage: python main.py predict <artifacts_path> <bert_model_path> <recipes_path> <predictions_output_path>"
        )
        return

    import predict

    _ = predict.main(args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    import logging
    import os

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    arguments = sys.argv[1:]

    if len(arguments) == 0:
        _print_help(sys.argv[0])
        sys.exit(1)

    operation = arguments[0]

    if operation == "train":
        _train(arguments[1:])
    elif operation == "predict":
        _predict(arguments[1:])
    elif operation == "help":
        _print_help(sys.argv[0])
    else:
        print(f"Unknown operation: {operation}")
        _print_help(sys.argv[0])
