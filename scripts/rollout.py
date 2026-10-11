from alc_video.inference.cli import parse_rollout_args, rollout_request
from alc_video.runtime.cli import initialize_cli
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import run_cli


def main(argv=None):
    args = parse_rollout_args(argv)
    initialize_cli(args.log_level)
    with input_errors():
        request = rollout_request(args)
    from alc_video.inference.rollout.pipeline import run_rollout

    run_rollout(request, resume=args.resume, log_level=args.log_level)


if __name__ == "__main__":
    run_cli(main)
