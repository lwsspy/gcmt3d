"""
Prints the name of a CMTSOLUTION from the eventname field.

Usage: 

    ioi-get-eventname <filename>


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.02.28 16.50

"""


from sys import argv
from lwsspy.seismo.source import CMTSource


def bin():

    if len(argv) != 1+1:
        print(__doc__)
    else:
        print(CMTSource.from_CMTSOLUTION_file(argv[1]))



