import argparse
import datetime
import logging
import pathlib
import sys
import typing

import yaml


def is_valid_file(path: str) -> str:
    """Identity function (i.e., the input is passed through and is not modified)
    that checks whether the given path is a valid file or not, raising an
    argparse.ArgumentTypeError if not valid.

    Parameters
    ----------
    path : str
        String representing a path to a file.

    Returns
    -------
    str
        The same string as in input

    Raises
    ------
    argparse.ArgumentTypeError
        An exception is raised if the given string does not represent a valid
        path to an existing file.
    """
    file = pathlib.Path(path)
    if not file.is_file():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path


def get_parser(description: str) -> argparse.ArgumentParser:
    """Function that generates the argument parser for the processor. Here, all
    the arguments and help message are defined.

    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.

    Returns
    -------
    argparse.ArgumentParser
        The created argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        "-c",
        dest="config_file_path",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file.",
    )
    return parser


def logged_main(description: str, main_fn: typing.Callable):
    """Function that wraps around your main function adding logging capabilities
    and basic configuration with a yaml file passed with `-c <config_file_path>`

    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.
    main_fn : typing.Callable
        Main function to execute
    """
    start_time = datetime.datetime.now()

    parser = get_parser(description)
    args = parser.parse_args()

    config_file = args.config_file_path
    with open(config_file) as yaml_file:
        config = yaml.full_load(yaml_file)

    log_folder = config["log_dir"]
    log_folder = pathlib.Path(log_folder)
    log_folder.mkdir(exist_ok=True, parents=True)
    log_level = config["log_level"]

    logging.getLogger().setLevel(logging.NOTSET)
    log_filename = (
        log_folder
        / f"{main_fn.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.getLevelName(log_level.upper()))
    frmttr_console = logging.Formatter(
        "%(asctime)s [%(name)s-%(levelname)s]: %(message)s"
    )
    console_handler.setFormatter(frmttr_console)
    logging.getLogger().addHandler(console_handler)

    logfile_handler = logging.FileHandler(filename=log_filename)
    logfile_handler.setLevel(logging.DEBUG)
    frmttr_logfile = logging.Formatter(
        "%(asctime)s [%(name)s-%(levelname)s]: %(message)s"
    )
    logfile_handler.setFormatter(frmttr_logfile)
    logging.getLogger().addHandler(logfile_handler)

    max_key_length = max(len(str(key)) for key in config.keys())
    aligned_output = ""
    for key, value in config.items():
        aligned_output += f"{key:<{max_key_length}} : {value}\n"

    logging.info("Configuration Complete.")
    logging.info("YAML Config File:\n%s", aligned_output)
    config["yaml_filepath"] = config_file
    main_fn(**config)

    logging.info(f"Close all. Execution time: {datetime.datetime.now() - start_time}")
    logging.getLogger().handlers.clear()
    logging.shutdown()
