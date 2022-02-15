import os


def write_status(statdir, message):

    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "w") as f:
        f.write(message)


def write_log(logdir, message):
    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "a") as f:
        f.write(message + "\n")


def clear_log(logdir):
    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "w") as f:
        f.close()
