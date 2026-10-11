from alc_video.inference.cli import inference_request, parse_inference_args
from alc_video.runtime.cli import initialize_cli
from alc_video.utils.errors import input_errors
from alc_video.utils.logging import run_cli


def main(argv=None):
    args = parse_inference_args(argv)
    initialize_cli(args.log_level)
    with input_errors():
        request = inference_request(args)
    from alc_video.inference.pipeline import run_inference

    run_inference(request, log_level=args.log_level)


if __name__ == "__main__":
    run_cli(main)
