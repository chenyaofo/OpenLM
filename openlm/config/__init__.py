
import typing
import sys
import pathlib
from dataclasses import dataclass

from pyhocon import ConfigFactory, ConfigTree

from torch4x.typed_args import TypedArgs, add_argument


def apply_modifications(modifications: typing.Sequence[str], conf: ConfigTree):
    special_cases = {
        "true": True,
        "false": False
    }
    if modifications is not None:
        for modification in modifications:
            key, value = modification.split("=")
            if value in special_cases.keys():
                eval_value = special_cases[value]
            else:
                try:
                    eval_value = eval(value)
                except Exception:
                    eval_value = value
            if key not in conf:
                raise ValueError(f"Key '{key}'' is not in the config tree!")
            conf.put(key, eval_value)
    return conf


@dataclass
class Args(TypedArgs):
    outdir: typing.Optional[str] = add_argument("-o", default=None)
    conf: str = add_argument("-c", default="")
    modifications: typing.Sequence[str] = add_argument("-M", nargs='+', help="list")


def get_args(argv=sys.argv):
    args, _ = Args.from_known_args(argv)
    args: Args

    args.outdir = pathlib.Path(args.outdir)

    args.conf: ConfigTree = ConfigFactory.parse_file(args.conf)
    args.outdir.mkdir(parents=True, exist_ok=True)

    apply_modifications(modifications=args.modifications, conf=args.conf)

    return args
