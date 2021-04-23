"""Module for handling snippet information"""
import re
from typing import List

from intervaltree import IntervalTree

WORDED = re.compile(r'\w')


def count_lines_length(snippet):
    lines = snippet.split(' / ')
    while WORDED.search(lines[-1]) is None:
        lines.pop()
    return len(lines)


class PhrasesTracker:
    """Keeps track of phrases for a given work"""

    def __init__(self, phrases):
        """

        ``phrases`` should be a list of dictionaries, where each dictionary has
        the following keys:
            * "snippet"
            * "tags"

        The "snippet" key should be associated with a string, indicating the
        text from a work.

        The "tags" key should be associated with a list of strings, indicating
        the loci over which the phrase extends. A few assumptions:
            * the list of loci are in sorted order, from earlier to later
            * the loci all take the form "<book number>.<line number>"
            * the line numbers are contiguous (i.e., no 495a, 495b)
        """
        self.book_to_intervals = _create_book_to_intervals(phrases)

    def find(self, locus) -> List[str]:
        """Finds the snippet(s) associated with the locus"""
        book_str, line_str = locus.split('.')
        line = int(line_str)
        return [a.data for a in self.book_to_intervals[book_str][line]]


def _create_book_to_intervals(phrases):
    book_to_intervals = {}
    for phrase in phrases:
        begin_locus = phrase['tags'][0]
        book_str, line_str = begin_locus.split('.')
        begin_line = int(line_str)
        interval_end = begin_line + len(phrase['tags'])
        if book_str not in book_to_intervals:
            book_to_intervals[book_str] = IntervalTree()
        cur_tree = book_to_intervals[book_str]
        cur_tree[begin_line:interval_end] = phrase['snippet']
    return book_to_intervals
