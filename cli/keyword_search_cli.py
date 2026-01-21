#!/usr/bin/env python3

import argparse
import json
import sys

from lib.keyword_search import build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            try:
                results = search_command(args.query)
            except FileNotFoundError as e:
                print(str(e))
                sys.exit(1)
            for i, result in enumerate(results, start=1):
                print(f"{i}. ({result['id']}) {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()