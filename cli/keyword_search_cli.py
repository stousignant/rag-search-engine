#!/usr/bin/env python3

import argparse
import json
import sys

from lib.keyword_search import build_command, idf_command, search_command, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

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
        case "tf":
            try:
                count = tf_command(args.doc_id, args.term)
            except FileNotFoundError as e:
                print(str(e))
                sys.exit(1)
            except ValueError as e:
                print(str(e))
                sys.exit(1)
            print(count)
        case "idf":
            try:
                idf = idf_command(args.term)
            except FileNotFoundError as e:
                print(str(e))
                sys.exit(1)
            except ValueError as e:
                print(str(e))
                sys.exit(1)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                tf_idf = tfidf_command(args.doc_id, args.term)
            except FileNotFoundError as e:
                print(str(e))
                sys.exit(1)
            except ValueError as e:
                print(str(e))
                sys.exit(1)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()