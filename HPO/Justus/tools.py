import sys
import time


def progress(iterable, text=None, inner=None, timed=None):
    """
    Generator for timed for loops with progress bar

    :param iterable, inner: iterable for outer and (optional) inner loop
    :param text: (optional) Task description
    :param timed: [list of (delta, f)] events that are triggered by calling <f> after <delta> seconds have passed
    """
    text = text + ' ' if text is not None else ''
    start = time.time()
    last = start
    if timed is not None:
        last_timed = {item: start for item in timed}

    def handle_events(force=False):
        for (dt, f), lt in last_timed.items():
            if now - lt > dt or force:
                f()
                last_timed[(dt, f)] = now

    # for loop
    if inner is None:
        for i, x in enumerate(iterable):
            now = time.time()
            if i == 0 or i == len(iterable) - 1 or now - last > 0.5:
                last = now
                # Progress percentage at step completion, TBD percentage shortly after step start
                perc = (i + 1) / len(iterable)
                inv_perc = len(iterable) / (i + 0.1)
                sys.stdout.write("\r%s[%.1f %%] - %d / %d - %.1fs [TBD: %.1fs]" %
                                 (text, 100 * perc, i + 1, len(iterable), now-start, (now-start) * inv_perc))
                sys.stdout.flush()
            # Call events
            if timed is not None:
                handle_events()
            yield x
    # for loop in for loop
    else:
        for i, x in enumerate(iterable):
            for j, y in enumerate(inner):
                now = time.time()
                if j == 0 or j == len(inner) - 1 or now - last > 0.5:
                    last = now
                    perc = (i * len(inner) + j + 1) / (len(iterable) * len(inner))
                    inv_perc = (len(iterable) * len(inner)) / (i * len(inner) + j + 0.1)
                    sys.stdout.write("\r%s[%.1f %%] - %d / %d (%d / %d) - %.1fs [TBD: %.1fs]" %
                                     (text, 100 * perc, i + 1, len(iterable), j + 1, len(inner),
                                      now-start, (now-start) * inv_perc))
                    sys.stdout.flush()
                # Call events
                if timed is not None:
                    handle_events()
                yield x, y

    if timed is not None:
        handle_events(force=True)
    print()
