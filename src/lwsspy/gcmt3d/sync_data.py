import os
import sys
import time
import platform
import asyncio
from typing import List, Optional
from random import randint


async def longprocess():

    """Run command in subprocess (shell)

    Note:
        This can be used if you wish to execute e.g. "copy"
        on Windows, which can only be executed in the shell.
    """
    seconds = randint(1, 14)
    command = f"echo Sleeping {seconds} s && sleep {seconds}"
    # Create subprocess
    process = await asyncio.create_subprocess_shell(
        command,
        stderr=asyncio.subprocess.PIPE)

    # Status
    print('Started:', command, '(pid = ' + str(process.pid) + ')')

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Progress
    if process.returncode == 0:
        print('Done:', command, '(pid = ' + str(process.pid) + ')')
    else:
        print('Failed:', command, '(pid = ' + str(process.pid) + ')')

    # Result
    result = stderr.decode().strip()

    # Real time print
    print(result)

    # Return stdout
    return result


async def run_command_shell(command):
    """Run command in subprocess (shell)

    Note:
        This can be used if you wish to execute e.g. "copy"
        on Windows, which can only be executed in the shell.
    """
    # Create subprocess
    process = await asyncio.create_subprocess_shell(
        command,
        stderr=asyncio.subprocess.PIPE)

    # Status
    print('Started:', command, '(pid = ' + str(process.pid) + ')')

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Progress
    if process.returncode == 0:
        print('Done:', command, '(pid = ' + str(process.pid) + ')')
    else:
        print('Failed:', command, '(pid = ' + str(process.pid) + ')')

    # Result
    result = stderr.decode().strip()

    # Real time print
    print(result)

    # Return stdout
    return result


def make_chunks(l, n):
    """Yield successive n-sized chunks from l.

    Note:
        Taken from https://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def run_asyncio_commands(tasks, max_concurrent_tasks=0):
    """Run tasks asynchronously using asyncio and return results

    If max_concurrent_tasks are set to 0, no limit is applied.

    Note:
        By default, Windows uses SelectorEventLoop, which does not support
        subprocesses. Therefore ProactorEventLoop is used on Windows.
        https://docs.python.org/3/library/asyncio-eventloops.html#windows
    """

    all_results = []

    if max_concurrent_tasks == 0:
        chunks = [tasks]
    else:
        chunks = make_chunks(l=tasks, n=max_concurrent_tasks)

    for tasks_in_chunk in chunks:
        if platform.system() == 'Windows':
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()

        commands = asyncio.gather(*tasks_in_chunk)  # Unpack list using *
        results = loop.run_until_complete(commands)
        all_results += results
        loop.close()
    return all_results


def sync_data(
        data_database: str, new_database: str,
        eventlist: Optional[List[str]] = None,
        n: int = 50):

    start = time.time()

    # If no list is provided, the entire database will be synchronized
    if eventlist is None:
        eventlist = os.listdir(data_database)
        # Initialize list of process
    processes = []

    # Rsync command to synchronize the databases in terms of events
    rsyncstr = 'rsync -a --include="*/" ' \
        '--include="*.mseed" ' \
        '--include="*.xml" ' \
        '--exclude="*"'

    # define processes
    print("[INFO] Starting event list...")

    # Don't run more than simultaneous jobs below
    commands = []
    for event in eventlist:

        # Full RSYNC command
        command = f"{rsyncstr} {data_database}/{event}/ {new_database}/{event}"

        # Create task for asyncio
        commands.append(command)

    # Create list of asynchronous tasks.
    tasks = []
    for command in commands:
        tasks.append(run_command_shell(command))

    # At most max_concurrent_tasks parallel tasks
    results = run_asyncio_commands(
        tasks, max_concurrent_tasks=n)
    print("[INFO] ... finished event list")

    end = time.time()
    rounded_end = ('{0:.4f}'.format(round(end-start, 4)))
    print('Script ran in about', str(rounded_end), 'seconds')


def bin():

    import argparse

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='data_database', type=str,
        help='Database that contains the downloaded data.')
    parser.add_argument(
        dest='new_database', help='Database for inversion', type=str)
    parser.add_argument(
        '-e', '--event-list', dest='event_list', nargs='+',
        help='List of events to sync', default=None,
        required=False)
    parser.add_argument(
        '-n', '--max-threads', dest='threads', type=int,
        help='Maximum number of concurrent tasks', default=20,
        required=False)
    args = parser.parse_args()

    sync_data(
        args.data_database, args.new_database, args.event_list, n=args.threads)
