import scripting
import training


def main(**config):
    config.pop("yaml_filepath", None)
    return training.run_from_config(**config)


if __name__ == "__main__":
    scripting.logged_main(
        "Train and optionally tune the SAHC Lightning implementation",
        main,
    )
