"""Allow running the reference decoder as: python -m reference_decoder"""
from reference_decoder.cli import _build_parser, main
main(_build_parser().parse_args())
