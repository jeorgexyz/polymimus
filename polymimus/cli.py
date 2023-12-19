from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .transcribe import TranscriptionResult, load_model, transcribe
from .mic import MicTranscriber

app = typer.Typer(name="polymimus", help="Local multilingual speech transcription.", add_completion=False)
console = Console()


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_srt(result: TranscriptionResult) -> str:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        start = _fmt_time(seg.start).replace(".", ",")
        end = _fmt_time(seg.end).replace(".", ",")
        lines += [str(i), f"{start} --> {end}", seg.text, ""]
    return "\n".join(lines)


def _to_txt(result: TranscriptionResult) -> str:
    return "\n".join(seg.text for seg in result.segments)


def _print_result(result: TranscriptionResult) -> None:
    console.print(
        Panel(
            f"[bold]Language:[/bold] {result.language.upper()} "
            f"([dim]{result.language_probability:.0%} confidence[/dim])  "
            f"[bold]Duration:[/bold] {_fmt_time(result.duration)}",
            title="[bold green]PolyMimus[/bold green]",
        )
    )

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Start", style="dim", width=14)
    table.add_column("End", style="dim", width=14)
    table.add_column("Text")

    for seg in result.segments:
        table.add_row(_fmt_time(seg.start), _fmt_time(seg.end), seg.text)

    console.print(table)


@app.command()
def transcribe_cmd(
    audio: Annotated[Path, typer.Argument(help="Audio file to transcribe")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model size: tiny, base, small, medium, large")] = "base",
    translate: Annotated[bool, typer.Option("--translate", "-t", help="Translate output to English")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Save transcript to .txt or .srt")] = None,
) -> None:
    """Transcribe an audio file."""
    if not audio.exists():
        console.print(f"[red]Error:[/red] File not found: {audio}")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task_id = progress.add_task(f"Loading [cyan]{model}[/cyan] model...", total=None)
        whisper_model = load_model(model)
        progress.update(task_id, description=f"Transcribing [cyan]{audio.name}[/cyan]...")
        start = time.perf_counter()
        result = transcribe(whisper_model, audio, translate=translate)
        elapsed = time.perf_counter() - start

    console.print(f"[dim]Finished in {elapsed:.1f}s[/dim]\n")
    _print_result(result)

    if output:
        content = _to_srt(result) if output.suffix.lower() == ".srt" else _to_txt(result)
        output.write_text(content, encoding="utf-8")
        console.print(f"\n[green]Saved to[/green] {output}")


@app.command()
def listen(
    model: Annotated[str, typer.Option("--model", "-m", help="Model size: tiny, base, small, medium, large")] = "base",
    translate: Annotated[bool, typer.Option("--translate", "-t", help="Translate output to English")] = False,
    aggressiveness: Annotated[int, typer.Option("--aggressiveness", "-a", help="VAD sensitivity 0-3 (higher = stricter)")] = 2,
    silence: Annotated[float, typer.Option("--silence", "-s", help="Seconds of silence before cutting a segment")] = 0.6,
) -> None:
    """Transcribe live microphone input."""
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task_id = progress.add_task(f"Loading [cyan]{model}[/cyan] model...", total=None)
        whisper_model = load_model(model)
        progress.update(task_id, description="Ready.")

    console.print(
        Panel(
            "Listening... Press [bold]Ctrl+C[/bold] to stop.",
            title="[bold green]PolyMimus[/bold green]",
        )
    )

    transcriber = MicTranscriber(
        model=whisper_model,
        aggressiveness=aggressiveness,
        silence_duration=silence,
        translate=translate,
    )

    def on_transcript(text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{ts}[/dim] [cyan]▸[/cyan] {text}")

    try:
        transcriber.run(on_transcript)
    except KeyboardInterrupt:
        transcriber.stop()
        console.print("\n[dim]Stopped.[/dim]")


# rename for CLI help display
transcribe_cmd.name = "transcribe"
