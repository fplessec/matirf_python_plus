"""
The benchmark, from the command line.

    python -m benchmarks run           # start (or resume) the full-resolution campaign
    python -m benchmarks status        # how far it got, by status
    python -m benchmarks stop          # ask it to stop cleanly before the next run
    python -m benchmarks specs         # print the matrix without running anything

Once installed (`pip install -e .`) the same commands are `benchmark run | status | stop`.

Safe and resumable by construction (benchmarks/runner.py):
  > each run is an isolated process with a time limit; a crash/hang is contained.
  > every finished run is saved atomically; `run` skips what is done and resumes.
  > INTERRUPT it three ways, all safe: Ctrl-C here (foreground), `... stop` from anywhere
    (writes a STOP file the loop honours before the next run), or — if Claude launched it —
    ask Claude to stop it. A run killed mid-way is redone cleanly on the next `run`.

Options: --dir <path> (default benchmarks/results/main), --timeout <s> (default 3600),
--stop-on-reject (halt on a realism-rejected run to investigate; by default such a run is
recorded as a data point and the campaign keeps going).
"""
import sys
from pathlib import Path

DEFAULT_DIR = Path("benchmarks/results/main")


def _opt(args, name, cast=str, default=None):
    if name in args:
        return cast(args[args.index(name) + 1])
    return default


def _cmd_run(args):
    from benchmarks.campaign import specs, summary
    from benchmarks.runner import run_campaign, CampaignStopped

    directory = Path(_opt(args, "--dir", str, str(DEFAULT_DIR)))
    timeout = _opt(args, "--timeout", float, 3600.0)
    ## atlas default: a realism-rejected run is a data point (method X unrealistic on
    ## structure Y), so keep going and record it; --stop-on-reject halts to investigate.
    stop_on_reject = "--stop-on-reject" in args
    accept_rejected = not stop_on_reject
    print(summary())
    print(f"Directory: {directory}   timeout/run: {timeout:.0f}s   "
          f"{'(stops on a rejected run)' if stop_on_reject else '(records rejected runs, keeps going)'}")
    print("Ctrl-C to interrupt, or `python -m benchmarks stop` from another terminal.\n")
    try:
        counts = run_campaign(specs(), directory, timeout=timeout,
                              stop_on_reject=not accept_rejected, accept_rejected=accept_rejected)
        print(f"\nDone this pass: {counts}")
    except CampaignStopped as stop:
        print(f"\n{stop}")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted. Relaunch `python -m benchmarks run` to resume.")
        sys.exit(130)


def _cmd_stop(args):
    directory = Path(_opt(args, "--dir", str, str(DEFAULT_DIR)))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "STOP").write_text("stop requested\n")
    print(f"STOP written to {directory / 'STOP'}. The campaign will exit before its next run.\n"
          f"Relaunch `python -m benchmarks run` to resume (it clears the flag).")


def _cmd_status(args):
    from benchmarks.campaign import specs
    from benchmarks.runner import campaign_records

    directory = Path(_opt(args, "--dir", str, str(DEFAULT_DIR)))
    planned = {spec.run_id for spec in specs()}
    records = campaign_records(directory)
    counts = {}
    for record in records:
        counts[record.get("status", "?")] = counts.get(record.get("status", "?"), 0) + 1
    done = counts.get("done", 0) + counts.get("rejected", 0)
    print(f"Campaign {directory}")
    print(f"  planned runs : {len(planned)}")
    print(f"  recorded     : {len(records)}  ->  {counts or '(none yet)'}")
    print(f"  remaining    : {max(0, len(planned) - done)}")
    if (directory / "STOP").exists():
        print("  STOP flag is set — a running campaign will stop before its next run.")


def _cmd_specs(args):
    from benchmarks.campaign import summary
    print(summary())


def main():
    commands = {"run": _cmd_run, "resume": _cmd_run, "status": _cmd_status,
                "stop": _cmd_stop, "specs": _cmd_specs}
    args = sys.argv[1:]
    command = args[0] if args else "status"
    if command not in commands:
        print(__doc__)
        sys.exit(1)
    commands[command](args[1:])


if __name__ == "__main__":
    main()
