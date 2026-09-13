#!/usr/bin/env python3
"""Compile the ActSim LaTeX manual (docs/manual) into a PDF.

The manual is a single ``report`` document, ``actsim_manual.tex``, that pulls in
one ``.tex`` file per chapter via ``\\input{sections/<name>}``.  The chapter
files are stored flat in ``docs/manual`` rather than in a ``sections/``
sub-directory, so this script stages them under the path the main file expects
before invoking the TeX engine.  Nothing in ``docs/manual`` is modified; all
intermediate files go to a build directory and only the finished PDF is copied
back.

Typical use (from anywhere in the repository)::

    python scripts/build_manual.py              # -> docs/manual/actsim_manual.pdf
    python scripts/build_manual.py --clean      # remove the build directory
    python scripts/build_manual.py --engine xelatex --output /tmp/manual.pdf

Requirements: a TeX distribution (TeX Live, MiKTeX, MacTeX) providing the
chosen engine (``pdflatex`` by default) on ``PATH``.  ``bibtex`` and
``makeindex`` are run automatically when the document asks for them.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "docs" / "manual"
DEFAULT_MAIN = "actsim_manual.tex"
BUILD_DIR_NAME = ".build"

ENGINES = ("pdflatex", "xelatex", "lualatex")

# Messages LaTeX prints when another pass is needed.
RERUN_PATTERNS = (
    re.compile(r"Rerun to get cross-references right"),
    re.compile(r"Rerun to get outlines right"),
    re.compile(r"Rerun LaTeX"),
    re.compile(r"Label\(s\) may have changed"),
    re.compile(r"Table widths have changed\. Rerun LaTeX"),
    re.compile(r"Package rerunfilecheck Warning"),
)

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


class BuildError(RuntimeError):
    """Raised when the PDF cannot be produced."""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[build_manual] {msg}", flush=True)


def find_main_file(source_dir: Path, main: Optional[str]) -> Path:
    """Return the root ``.tex`` file, auto-detecting it when not given."""
    if main:
        path = Path(main)
        if not path.is_absolute():
            path = source_dir / path
        if not path.is_file():
            raise BuildError(f"main file not found: {path}")
        return path

    default = source_dir / DEFAULT_MAIN
    if default.is_file():
        return default

    candidates = [
        p for p in sorted(source_dir.glob("*.tex"))
        if "\\documentclass" in p.read_text(encoding="utf-8", errors="replace")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise BuildError(f"no .tex file with \\documentclass found in {source_dir}")
    names = ", ".join(c.name for c in candidates)
    raise BuildError(
        f"several root documents found ({names}); pick one with --main"
    )


def referenced_inputs(tex_file: Path) -> List[str]:
    """Return the raw arguments of every \\input / \\include in ``tex_file``."""
    text = tex_file.read_text(encoding="utf-8", errors="replace")
    # Drop comments so a commented-out \input is not staged or reported.
    text = re.sub(r"(?<!\\)%.*", "", text)
    return [m.strip() for m in INPUT_RE.findall(text)]


def stage_missing_inputs(main_file: Path, source_dir: Path, staging_dir: Path,
                         *, quiet: bool) -> int:
    """Create ``staging_dir/<relative path>.tex`` for inputs that don't resolve.

    The main file references ``sections/<chapter>`` but the chapter files live
    directly in ``source_dir``.  For each unresolved reference whose basename
    exists in ``source_dir``, a copy (symlink where supported) is placed at
    the expected relative path inside ``staging_dir``.  ``staging_dir`` is
    later prepended to ``TEXINPUTS`` so kpathsea finds the files there.

    Returns the number of staged files.  Raises ``BuildError`` if a reference
    cannot be satisfied at all.
    """
    staged = 0
    missing: List[str] = []
    for ref in referenced_inputs(main_file):
        rel = ref if ref.endswith(".tex") else ref + ".tex"
        if (source_dir / rel).is_file():
            continue  # resolves normally
        candidate = source_dir / Path(rel).name
        if not candidate.is_file():
            missing.append(ref)
            continue
        target = staging_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(candidate.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(candidate, target)
        staged += 1
    if missing:
        raise BuildError(
            "the following \\input/\\include targets could not be found in "
            f"{source_dir}: " + ", ".join(missing)
        )
    if staged:
        log(f"staged {staged} chapter file(s) under {staging_dir}", quiet=quiet)
    return staged


def prepare_include_dirs(main_file: Path, build_dir: Path) -> None:
    """``\\include{a/b}`` writes ``a/b.aux`` into the output directory; the
    sub-directory must already exist or the engine aborts."""
    for ref in referenced_inputs(main_file):
        parent = Path(ref).parent
        if str(parent) not in ("", "."):
            (build_dir / parent).mkdir(parents=True, exist_ok=True)


def tex_env(source_dir: Path, staging_dir: Path, build_dir: Path) -> dict:
    """Environment with TEXINPUTS/BIBINPUTS pointing at our directories.

    A trailing ``os.pathsep`` keeps the distribution's default search path.
    """
    env = os.environ.copy()
    extra = [str(source_dir), str(staging_dir), str(build_dir)]
    for var in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
        current = env.get(var, "")
        env[var] = os.pathsep.join(extra + ([current] if current else [""]))
    # Keep TeX from writing anything relative to the *source* tree.
    env.setdefault("max_print_line", "10000")
    return env


def run(cmd: Sequence[str], *, cwd: Path, env: dict, verbose: bool) -> subprocess.CompletedProcess:
    if verbose:
        print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=None if verbose else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )


def needs_rerun(log_text: str) -> bool:
    return any(p.search(log_text) for p in RERUN_PATTERNS)


def needs_bibtex(aux_file: Path) -> bool:
    if not aux_file.is_file():
        return False
    text = aux_file.read_text(encoding="utf-8", errors="replace")
    return "\\bibdata{" in text and "\\citation{" in text


def extract_errors(log_text: str, limit: int = 20) -> List[str]:
    """Pull the most useful lines out of a failed LaTeX log."""
    lines = log_text.splitlines()
    picked: List[str] = []
    for i, line in enumerate(lines):
        if line.startswith("!") or re.match(r"^.+:\d+: ", line):
            picked.append(line)
            # Include the context lines LaTeX prints right after an error.
            for extra in lines[i + 1:i + 4]:
                if extra.strip():
                    picked.append("    " + extra)
        if len(picked) >= limit:
            break
    return picked


# ----------------------------------------------------------------------------
# Main build routine
# ----------------------------------------------------------------------------
def build(source_dir: Path, main: Optional[str], engine: str, output: Optional[Path],
          build_dir: Optional[Path], max_passes: int, keep_build: bool,
          quiet: bool, verbose: bool) -> Path:
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise BuildError(f"source directory does not exist: {source_dir}")
    main_file = find_main_file(source_dir, main)
    job = main_file.stem

    if shutil.which(engine) is None:
        raise BuildError(
            f"'{engine}' not found on PATH; install a TeX distribution "
            "(e.g. TeX Live / MiKTeX) or choose another --engine"
        )

    build_dir = (build_dir or source_dir / BUILD_DIR_NAME).resolve()
    staging_dir = build_dir / "_staging"
    build_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    stage_missing_inputs(main_file, source_dir, staging_dir, quiet=quiet)
    prepare_include_dirs(main_file, build_dir)
    env = tex_env(source_dir, staging_dir, build_dir)

    engine_cmd = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={build_dir}",
        f"-jobname={job}",
        main_file.name,
    ]
    log_file = build_dir / f"{job}.log"
    aux_file = build_dir / f"{job}.aux"
    idx_file = build_dir / f"{job}.idx"
    pdf_file = build_dir / f"{job}.pdf"

    def latex_pass(n: int) -> str:
        log(f"{engine} pass {n}", quiet=quiet)
        proc = run(engine_cmd, cwd=source_dir, env=env, verbose=verbose)
        text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
        if proc.returncode != 0:
            errors = extract_errors(text) or (proc.stdout or "").splitlines()[-30:]
            raise BuildError(
                f"{engine} failed (exit {proc.returncode}). See {log_file}\n"
                + "\n".join(errors)
            )
        return text

    log_text = latex_pass(1)

    ran_aux_tool = False
    if needs_bibtex(aux_file):
        if shutil.which("bibtex") is None:
            raise BuildError("document uses a bibliography but 'bibtex' is not on PATH")
        log("bibtex", quiet=quiet)
        proc = run(["bibtex", job], cwd=build_dir, env=env, verbose=verbose)
        if proc.returncode != 0:
            raise BuildError("bibtex failed:\n" + (proc.stdout or ""))
        ran_aux_tool = True
    if idx_file.is_file() and idx_file.stat().st_size > 0:
        if shutil.which("makeindex") is None:
            raise BuildError("document uses an index but 'makeindex' is not on PATH")
        log("makeindex", quiet=quiet)
        proc = run(["makeindex", idx_file.name], cwd=build_dir, env=env, verbose=verbose)
        if proc.returncode != 0:
            raise BuildError("makeindex failed:\n" + (proc.stdout or ""))
        ran_aux_tool = True

    passes = 1
    while passes < max_passes and (ran_aux_tool or needs_rerun(log_text)):
        ran_aux_tool = False
        passes += 1
        log_text = latex_pass(passes)

    if needs_rerun(log_text):
        log(f"warning: cross-references may still be stale after {passes} passes "
            f"(raise --max-passes)", quiet=quiet)

    if not pdf_file.is_file():
        raise BuildError(f"{engine} reported success but {pdf_file} was not produced")

    output = (output or source_dir / f"{job}.pdf").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_file, output)

    undefined = len(re.findall(r"Reference `[^']+' on page \d+ undefined", log_text))
    if undefined:
        log(f"warning: {undefined} undefined reference(s); check {log_file}", quiet=quiet)

    if not keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)
    return output


def clean(source_dir: Path, build_dir: Optional[Path], quiet: bool) -> None:
    build_dir = (build_dir or source_dir / BUILD_DIR_NAME).resolve()
    if build_dir.exists():
        shutil.rmtree(build_dir)
        log(f"removed {build_dir}", quiet=quiet)
    else:
        log("nothing to clean", quiet=quiet)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the LaTeX manual in docs/manual into a PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="directory containing the .tex sources")
    parser.add_argument("--main", default=None,
                        help="root .tex file (relative to --source-dir); "
                             f"defaults to {DEFAULT_MAIN} or the only file with \\documentclass")
    parser.add_argument("--engine", choices=ENGINES, default="pdflatex",
                        help="TeX engine to run")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="where to write the PDF (default: <source-dir>/<main>.pdf)")
    parser.add_argument("--build-dir", type=Path, default=None,
                        help=f"directory for intermediate files (default: <source-dir>/{BUILD_DIR_NAME})")
    parser.add_argument("--max-passes", type=int, default=5,
                        help="upper bound on LaTeX runs used to settle cross-references")
    parser.add_argument("--keep-build", action="store_true",
                        help="keep the build directory (logs, .aux, ...) after a successful build")
    parser.add_argument("--clean", action="store_true",
                        help="delete the build directory and exit")
    parser.add_argument("--quiet", "-q", action="store_true", help="only print errors")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="stream the full engine output")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.clean:
            clean(args.source_dir.resolve(), args.build_dir, args.quiet)
            return 0
        pdf = build(
            source_dir=args.source_dir,
            main=args.main,
            engine=args.engine,
            output=args.output,
            build_dir=args.build_dir,
            max_passes=max(1, args.max_passes),
            keep_build=args.keep_build,
            quiet=args.quiet,
            verbose=args.verbose,
        )
    except BuildError as exc:
        print(f"[build_manual] error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[build_manual] interrupted", file=sys.stderr)
        return 130
    log(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KiB)", quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
