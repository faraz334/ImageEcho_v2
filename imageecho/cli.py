import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich         import print as rprint
from tqdm         import tqdm

from imageecho.engines import (
    FgsmEngine, PgdEngine, LsbEngine, DctEngine,
    CwEngine, DeepFoolEngine, AutoPgdEngine,
    PatchEngine, GaussianEngine, JsmaEngine
)
from imageecho.context import EchoContext

console = Console()

ENGINE_MAP = {
    "fgsm":      FgsmEngine,
    "pgd":       PgdEngine,
    "lsb":       LsbEngine,
    "dct":       DctEngine,
    "cw":        CwEngine,
    "deepfool":  DeepFoolEngine,
    "auto_pgd":  AutoPgdEngine,
    "patch":     PatchEngine,
    "gaussian":  GaussianEngine,
    "jsma":      JsmaEngine,
}


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_image(img: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def print_report(report, elapsed: float):
    fooled_color = "green" if report.fooled else "red"
    fooled_text  = "FOOLED" if report.fooled else "SAME CLASS"

    t = Table(show_header=True, header_style="bold #7777cc",
              border_style="#333355", expand=False)
    t.add_column("Metric",         style="#888899", width=18)
    t.add_column("Value",          style="#eeeeff", width=20)

    t.add_row("Engine",            report.engine_name)
    t.add_row("Epsilon",           f"{report.epsilon:.5f}  ({round(report.epsilon*255)}/255)")
    t.add_row("SSIM",              f"{report.ssim:.4f}")
    t.add_row("PSNR",              f"{report.psnr:.1f} dB")
    t.add_row("Mean ?",            f"{report.mean_delta:.3f}")
    t.add_row("Max ?",             f"{report.max_delta:.1f}")
    t.add_row("Pixels Altered",    f"{report.pixels_altered:,}")
    t.add_row("Original Class",    str(report.original_class))
    t.add_row("Perturbed Class",   str(report.perturbed_class))
    t.add_row("Confidence Before", f"{report.original_confidence:.3f}")
    t.add_row("Confidence After",  f"{report.perturbed_confidence:.3f}")
    t.add_row("Result",            f"[{fooled_color}]{fooled_text}[/{fooled_color}]")
    t.add_row("Time",              f"{elapsed:.2f}s")

    console.print(t)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_run(args):
    """Run a single attack on one image."""
    if not Path(args.image).exists():
        console.print(f"[red]Error:[/red] image not found: {args.image}")
        sys.exit(1)

    if args.engine not in ENGINE_MAP:
        console.print(f"[red]Error:[/red] unknown engine '{args.engine}'")
        console.print(f"Available: {', '.join(ENGINE_MAP)}")
        sys.exit(1)

    console.print(Panel(
        f"[bold #7777cc]ImageEcho[/bold #7777cc]  —  "
        f"engine=[bold]{args.engine}[/bold]  "
        f"e={args.epsilon}/255",
        expand=False
    ))

    image   = load_image(args.image)
    epsilon = args.epsilon / 255.0
    engine  = ENGINE_MAP[args.engine](epsilon=epsilon)
    ctx     = EchoContext(engine)

    with console.status(f"Running [bold]{args.engine}[/bold] attack..."):
        t0              = time.time()
        adv, report     = ctx.run(image)
        elapsed         = time.time() - t0

    print_report(report, elapsed)

    out = args.output or f"adv_{args.engine}.png"
    save_image(adv, out)
    console.print(f"\n[green]Saved ?[/green] {out}")


def cmd_benchmark(args):
    """Run all 10 engines and print comparison table."""
    if not Path(args.image).exists():
        console.print(f"[red]Error:[/red] image not found: {args.image}")
        sys.exit(1)

    console.print(Panel(
        f"[bold #7777cc]ImageEcho Benchmark[/bold #7777cc]  —  "
        f"e={args.epsilon}/255  —  all 10 engines",
        expand=False
    ))

    image   = load_image(args.image)
    epsilon = args.epsilon / 255.0
    results = []

    for name in tqdm(ENGINE_MAP, desc="Running engines", unit="engine"):
        try:
            t0          = time.time()
            engine      = ENGINE_MAP[name](epsilon=epsilon)
            ctx         = EchoContext(engine)
            _, report   = ctx.run(image)
            elapsed     = time.time() - t0
            results.append((name, report, elapsed))
        except Exception as e:
            console.print(f"[red]  {name} failed:[/red] {e}")

    # Print summary table
    t = Table(
        title="Benchmark Results",
        show_header=True,
        header_style="bold #7777cc",
        border_style="#333355",
        expand=True
    )
    t.add_column("Engine",          style="#aaaacc", width=12)
    t.add_column("SSIM",            style="#eeeeff", width=8)
    t.add_column("PSNR (dB)",       style="#eeeeff", width=10)
    t.add_column("Mean ?",          style="#eeeeff", width=8)
    t.add_column("Pixels Altered",  style="#eeeeff", width=16)
    t.add_column("Fooled",          width=10)
    t.add_column("Time (s)",        style="#888899", width=10)

    for name, r, elapsed in results:
        fc = "green" if r.fooled else "red"
        ft = "YES" if r.fooled else "NO"
        t.add_row(
            name,
            f"{r.ssim:.4f}",
            f"{r.psnr:.1f}",
            f"{r.mean_delta:.2f}",
            f"{r.pixels_altered:,}",
            f"[{fc}]{ft}[/{fc}]",
            f"{elapsed:.1f}s",
        )

    console.print(t)

    fooled_count = sum(1 for _, r, _ in results if r.fooled)
    console.print(
        f"\n[bold]Summary:[/bold] {fooled_count}/{len(results)} engines "
        f"successfully fooled the classifier"
    )

    # Save markdown report if requested
    if args.save_report:
        lines = ["# ImageEcho Benchmark Report\n\n"]
        lines.append("| Engine | SSIM | PSNR | Mean ? | Fooled |\n")
        lines.append("|--------|------|------|--------|--------|\n")
        for name, r, _ in results:
            lines.append(
                f"| {name} | {r.ssim:.4f} | {r.psnr:.1f} | "
                f"{r.mean_delta:.2f} | {'YES' if r.fooled else 'NO'} |\n"
            )
        Path("benchmark_report.md").write_text("".join(lines), encoding='utf-8')
        console.print("[green]Report saved ?[/green] benchmark_report.md")


def cmd_batch(args):
    """Run an engine on all images in a folder."""
    folder = Path(args.folder)
    if not folder.exists():
        console.print(f"[red]Error:[/red] folder not found: {args.folder}")
        sys.exit(1)

    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in exts]

    if not images:
        console.print(f"[red]No images found in {args.folder}[/red]")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilon = args.epsilon / 255.0

    engines_to_run = (
        list(ENGINE_MAP.keys()) if args.engine == "all"
        else [args.engine]
    )

    console.print(Panel(
        f"[bold #7777cc]ImageEcho Batch[/bold #7777cc]  —  "
        f"{len(images)} images  ×  {len(engines_to_run)} engine(s)  —  "
        f"e={args.epsilon}/255",
        expand=False
    ))

    total = len(images) * len(engines_to_run)

    with tqdm(total=total, desc="Processing", unit="attack") as pbar:
        for img_path in images:
            image = load_image(str(img_path))
            for eng_name in engines_to_run:
                try:
                    engine     = ENGINE_MAP[eng_name](epsilon=epsilon)
                    ctx        = EchoContext(engine)
                    adv, _     = ctx.run(image)
                    out_name   = f"{img_path.stem}_{eng_name}{img_path.suffix}"
                    save_image(adv, str(out_dir / out_name))
                except Exception as e:
                    console.print(f"[red]  {img_path.name} / {eng_name}: {e}[/red]")
                pbar.update(1)

    console.print(
        f"\n[green]Done.[/green] Outputs saved to [bold]{out_dir}[/bold]"
    )


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imageecho",
        description="ImageEcho — Adversarial ML through invisible pixel perturbation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  imageecho run photo.png --engine fgsm --epsilon 8
  imageecho run photo.png --engine mi_fgsm --epsilon 16 --output out.png
  imageecho benchmark photo.png --epsilon 8 --save-report
  imageecho batch ./images/ --engine pgd --epsilon 8 --output ./perturbed/
  imageecho batch ./images/ --engine all --epsilon 8 --output ./perturbed/
        """
    )
    parser.add_argument("--version", action="version", version="ImageEcho 0.1.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single attack on one image")
    p_run.add_argument("image",                  help="Path to input image")
    p_run.add_argument("--engine", "-e",         default="fgsm",
                       choices=list(ENGINE_MAP.keys()),
                       help="Attack engine (default: fgsm)")
    p_run.add_argument("--epsilon", "-eps",      type=int, default=8,
                       help="Epsilon value x/255 (default: 8)")
    p_run.add_argument("--output", "-o",         default=None,
                       help="Output path (default: adv_<engine>.png)")

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Benchmark all 10 engines")
    p_bench.add_argument("image",                help="Path to input image")
    p_bench.add_argument("--epsilon", "-eps",    type=int, default=8,
                         help="Epsilon value x/255 (default: 8)")
    p_bench.add_argument("--save-report",        action="store_true",
                         help="Save results to benchmark_report.md")

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Process a folder of images")
    p_batch.add_argument("folder",               help="Input folder path")
    p_batch.add_argument("--engine", "-e",       default="fgsm",
                         help="Engine name or 'all' (default: fgsm)")
    p_batch.add_argument("--epsilon", "-eps",    type=int, default=8,
                         help="Epsilon value x/255 (default: 8)")
    p_batch.add_argument("--output", "-o",       default="perturbed",
                         help="Output folder (default: perturbed)")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()
