from typing import Dict, List, Tuple, Type, Any
import argparse
import sys


class Config:
    """
    Wrapper around argparse that makes things easier.
    """

    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> "Config":
        config = Config()
        for k, v in dict.items():
            setattr(config, k, v)
        return config

    @staticmethod
    def from_command_line(arg_defaults: List[Tuple[str, Type, Any, str]], description="") -> "Config":

        parser = argparse.ArgumentParser(description=description)

        for arg_key, arg_type, arg_val, arg_desc in arg_defaults:
            parser.add_argument(f"--{arg_key}", type=(arg_type if arg_type != list else str), default=arg_val, help=arg_desc, nargs=(None if arg_type != list else '+'))

        OPTS = Config()
        OPTS = parser.parse_args(namespace=OPTS)

        for k, v in OPTS:
            print(f" - ARG {k}: {v}", file=sys.stderr)

        return OPTS

    @staticmethod
    def nb_init(arg_defaults: List[Tuple[str, Type, Any, str]], description="") -> "Config":

        parser = argparse.ArgumentParser(description=description)

        for arg_key, arg_type, arg_val, arg_desc in arg_defaults:
            parser.add_argument(f"--{arg_key}", type=arg_type, default=arg_val, help=arg_desc)

        parser.add_argument('-f', type=str)


        OPTS = Config()
        OPTS = parser.parse_args(namespace=OPTS)

        for k, v in OPTS:
            print(f" - ARG {k}: {v}", file=sys.stderr)

        return OPTS