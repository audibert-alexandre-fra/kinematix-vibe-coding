"""Point d'entrée en ligne de commande (complété à l'étape 12)."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinematix")
    parser.parse_args(argv)
    return 0
