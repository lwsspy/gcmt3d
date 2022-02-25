import os


def write_status(outdir, message):

    statdir = outdir
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "w") as f:
        f.write(message)


def write_log(outdir, message):

    logdir = outdir
    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "a") as f:
        f.write(message + "\n")


def clear_log(outdir):

    logdir = outdir

    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "w") as f:
        f.close()
