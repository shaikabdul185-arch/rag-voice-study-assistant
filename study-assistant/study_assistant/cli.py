"""`study` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .db import KINDS, connect, init_schema, stats


def _embedder(settings):
    from .embeddings import get_embedder

    try:
        return get_embedder(settings.embed_backend, settings.embed_model)
    except Exception as exc:  # missing package, failed download, wrong dimension, ...
        raise SystemExit(
            f"Could not load embedding model {settings.embed_model!r} ({type(exc).__name__}: {exc}).\n"
            "The first run downloads it from Hugging Face; check your network, or pre-download it "
            "and set EMBED_MODEL to the local folder."
        ) from exc


def _agent(settings, conn, verbose: bool = True):
    import anthropic

    from .agent import StudyAgent
    from .retrieval import PgVectorRetriever
    from .tools import Toolbox

    retriever = PgVectorRetriever(conn, _embedder(settings))
    toolbox = Toolbox(retriever, anthropic.Anthropic(), settings.claude_model, settings.claude_effort,
                      use_fallbacks=settings.use_fallbacks)

    def on_tool(name: str, args: dict) -> None:
        if verbose:
            preview = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())
            print(f"  -> {name}({preview})", file=sys.stderr)

    return StudyAgent(toolbox, on_tool=on_tool)


def cmd_init_db(settings, args) -> int:
    with connect(settings.database_url) as conn:
        init_schema(conn)
    print("Schema ready.")
    return 0


def cmd_ingest(settings, args) -> int:
    from .ingest import ingest_file, iter_files

    files = iter_files([Path(p) for p in args.paths])
    if not files:
        print("No supported files (.pdf, .md, .txt) found.")
        return 1
    embedder = _embedder(settings)
    with connect(settings.database_url) as conn:
        for f in files:
            res = ingest_file(conn, embedder, f, args.kind, settings.chunk_size, settings.chunk_overlap)
            extra = f" ({res.chunks} chunks, {res.detail})" if res.status == "ingested" else (
                f" ({res.detail})" if res.detail else "")
            print(f"{res.status:>9}  {res.path}{extra}")
    return 0


def cmd_stats(settings, args) -> int:
    with connect(settings.database_url) as conn:
        rows = stats(conn)
    if not rows:
        print("Nothing indexed yet. Run `study ingest <folder>`.")
    for kind, docs, chunks in rows:
        print(f"{kind:<11} {docs:>4} documents  {chunks:>6} chunks")
    return 0


def cmd_search(settings, args) -> int:
    from .retrieval import PgVectorRetriever, format_hits

    with connect(settings.database_url) as conn:
        hits = PgVectorRetriever(conn, _embedder(settings)).search(
            args.query, k=args.k, kinds=[args.kind] if args.kind else None)
    print(format_hits(hits))
    return 0


def cmd_ask(settings, args) -> int:
    from .agent import friendly_api_error

    with connect(settings.database_url) as conn:
        agent = _agent(settings, conn)
        try:
            reply = agent.ask(" ".join(args.question))
        except Exception as exc:
            print(friendly_api_error(exc), file=sys.stderr)
            return 1
    print(reply.text)
    return 0


def cmd_chat(settings, args) -> int:
    from .agent import friendly_api_error

    with connect(settings.database_url) as conn:
        agent = _agent(settings, conn)
        print("Study chat. Commands: /reset, /quit")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not q:
                continue
            if q in ("/quit", "/exit"):
                return 0
            if q == "/reset":
                agent.reset()
                print("(conversation cleared)")
                continue
            try:
                print(f"\nassistant> {agent.ask(q).text}")
            except Exception as exc:
                print(friendly_api_error(exc), file=sys.stderr)


def cmd_voice(settings, args) -> int:
    from .voice import run_voice

    with connect(settings.database_url) as conn:
        run_voice(_agent(settings, conn), settings.whisper_model, args.file, tts=not args.no_tts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="study", description="RAG + voice study assistant")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the pgvector extension and tables").set_defaults(func=cmd_init_db)

    ing = sub.add_parser("ingest", help="index PDF/Markdown/text files or folders")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--kind", choices=KINDS,
                     help="collection; default inferred from folder (notes/, exams/, keys/)")
    ing.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="show what is indexed").set_defaults(func=cmd_stats)

    se = sub.add_parser("search", help="raw vector search (no LLM), for debugging retrieval")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--kind", choices=KINDS)
    se.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="interactive multi-turn chat").set_defaults(func=cmd_chat)

    v = sub.add_parser("voice", help="talk to the assistant (needs the [voice] extra)")
    v.add_argument("--file", help="transcribe this audio file instead of using the microphone")
    v.add_argument("--no-tts", action="store_true", help="print answers instead of speaking them")
    v.set_defaults(func=cmd_voice)
    return p


def main(argv: list[str] | None = None) -> int:
    import psycopg

    args = build_parser().parse_args(argv)
    settings = load_settings()
    try:
        return args.func(settings, args)
    except psycopg.OperationalError as exc:
        print(f"Could not connect to the database at DATABASE_URL.\n  {str(exc).strip().splitlines()[0]}\n"
              "Is it running? With Docker: `docker compose up -d` (from the study-assistant folder).",
              file=sys.stderr)
        return 1
    except psycopg.errors.UndefinedTable:
        print("The tables don't exist yet. Run `study init-db` first.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
