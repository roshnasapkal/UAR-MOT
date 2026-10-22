#!/usr/bin/env python3
"""Run the pinned YOLOX CLI with training exceptions propagated to the parent.

The upstream @logger.catch wrapper suppresses exceptions. Reuse its parser,
experiment loading and launch function, calling the wrapped training function
directly. No source files, training settings or evaluation logic are modified.
"""
import runpy

from colab_detector import YOLOX


def launch_entry(entry):
    args = entry["make_parser"]().parse_args()
    exp = entry["get_exp"](args.exp_file, args.name)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    num_gpu = entry["torch"].cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= entry["torch"].cuda.device_count()
    dist_url = "auto" if args.dist_url is None else args.dist_url
    # Loguru uses functools.wraps. Refuse an unexpected entry point rather than
    # accidentally retaining the exception-suppressing decorator.
    training_main = entry["main"].__wrapped__
    entry["launch"](
        training_main, num_gpu, args.num_machines, args.machine_rank,
        backend=args.dist_backend, dist_url=dist_url, args=(exp, args),
    )


if __name__ == "__main__":
    launch_entry(runpy.run_path(str(YOLOX / "tools/train.py")))
