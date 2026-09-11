def log(logfile: str, *args, **kwargs):
    """Prints to stdout and mirrors the exact output to the log file."""
    # Write to terminal
    print(*args, **kwargs)

    # Write to file
    with open(logfile, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)