#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_model, verify_embeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    subparsers.add_parser("verify_embeddings", help="Verify or build movie embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for a single text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()