# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/ichard26/pytest-balance-xdist/blob/main/NOTICE.txt
#
# History of changes:
#
# - Add (some) type annotations
# - Add autodiscovery of worker count
# - Switch to pytest's cache
# - Make show_worker_times() easier to use (it has a CLI now!)

import argparse
import collections
import csv
import os
import shutil
import time
from pathlib import Path
from typing import DefaultDict, Generator, Optional, Union

import pytest
import xdist.scheduler

__author__ = "Richard Si & Ned Batchelder"
__version__ = "1.0.0"


def pytest_addoption(parser):
    """Auto-called to define ini-file settings."""
    parser.addini(
        "balanced_clumps",
        type="linelist",
        help="Test substrings to assign to the same worker",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Registers our pytest plugin."""
    config.pluginmanager.register(BalanceXdistPlugin(config), "balance_xdist_plugin")


class BalanceXdistPlugin:
    """The plugin."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.running_all = self.config.getoption("-k") == ""
        self.times: DefaultDict[str, float] = collections.defaultdict(float)
        self.chunk_count = self.config.getoption("-n", 4)
        self.worker = os.environ.get("PYTEST_XDIST_WORKER", "none")
        self.timings_csv: Optional[Path] = None

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """Called once before any tests are run, but in every worker."""
        if not self.running_all:
            return

        assert session.config.cache is not None, "no cache?!"
        cache_dir = Path(session.config.cache.mkdir("balance-xdist"))
        self.timings_csv = Path(cache_dir, f"{self.worker}-timings.csv")

        if self.worker == "none" and cache_dir.exists():
            for csv_file in cache_dir.iterdir():
                with csv_file.open(newline="") as fcsv:
                    reader = csv.reader(fcsv)
                    for row in reader:
                        self.times[row[1]] += float(row[3])
            shutil.rmtree(cache_dir)

    def write_duration_row(self, item: pytest.Item, phase: str, duration: float) -> None:
        """Helper to write a row to the tracked-test csv file."""
        if not self.running_all:
            return

        assert self.timings_csv is not None
        with self.timings_csv.open("a", newline="") as fcsv:
            csv.writer(fcsv).writerow([self.worker, item.nodeid, phase, duration])

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> Generator[None, None, None]:
        """Run once for each test."""
        start = time.time()
        yield
        self.write_duration_row(item, "setup", time.time() - start)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        """Run once for each test."""
        start = time.time()
        yield
        self.write_duration_row(item, "call", time.time() - start)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item: pytest.Item) -> Generator[None, None, None]:
        """Run once for each test."""
        start = time.time()
        yield
        self.write_duration_row(item, "teardown", time.time() - start)

    @pytest.hookimpl(trylast=True)
    def pytest_xdist_make_scheduler(self, config, log):
        """Create our BalancedScheduler using time data from the last run."""
        # Assign tests to chunks
        totals = [0] * self.chunk_count
        tests = collections.defaultdict(set)

        # First put the difficult ones all in one worker
        clumped = set()
        clumps = config.getini("balanced_clumps")
        for i, clump_word in enumerate(clumps):
            clump_nodes = {nodeid for nodeid in self.times.keys() if clump_word in nodeid}
            i %= self.chunk_count
            tests[i].update(clump_nodes)
            totals[i] += sum(self.times[nodeid] for nodeid in clump_nodes)
            clumped.update(clump_nodes)

        # Then assign the rest in descending order
        rest = [(nodeid, t) for (nodeid, t) in self.times.items() if nodeid not in clumped]
        rest.sort(key=lambda item: item[1], reverse=True)
        for nodeid, t in rest:
            lightest = min(enumerate(totals), key=lambda pair: pair[1])[0]
            tests[lightest].add(nodeid)
            totals[lightest] += t

        test_chunks = {}
        for chunk_id, nodeids in tests.items():
            for nodeid in nodeids:
                test_chunks[nodeid] = chunk_id

        return BalancedScheduler(config, log, clumps, test_chunks)


class BalancedScheduler(xdist.scheduler.LoadScopeScheduling):
    """A balanced-chunk test scheduler for pytest-xdist."""

    def __init__(self, config, log, clumps, test_chunks) -> None:
        super().__init__(config, log)
        self.clumps = clumps
        self.test_chunks = test_chunks

    def _split_scope(self, nodeid):
        """Assign a chunk id to a test node."""
        # If we have a chunk assignment for this node, return it.
        scope = self.test_chunks.get(nodeid)
        if scope is not None:
            return scope

        # If this is a node that should be clumped, clump it.
        for i, clump_word in enumerate(self.clumps):
            if clump_word in nodeid:
                return f"clump{i}"

        # Otherwise every node is a separate chunk.
        return nodeid


def show_worker_times(tests_csv_dir: Union[str, os.PathLike]) -> None:
    """Ad-hoc utility to show data from the last tracked-test run."""
    times: DefaultDict[str, float] = collections.defaultdict(float)
    tests: DefaultDict[str, int] = collections.defaultdict(int)

    for csv_file in Path(tests_csv_dir).iterdir():
        with csv_file.open(newline="") as fcsv:
            reader = csv.reader(fcsv)
            for row in reader:
                worker = row[0]
                duration = float(row[3])
                times[worker] += duration
                if row[2] == "call":
                    tests[worker] += 1

    for worker in sorted(tests.keys()):
        print(f"{worker}: {tests[worker]:3d} {times[worker]:.2f}")

    total = sum(times.values())
    avg = total / len(times)
    print(f"total: {total:.2f}, avg: {avg:.2f}")
    lo = min(times.values())
    hi = max(times.values())
    print(f"lo = {lo:.2f}; hi = {hi:.2f}; gap = {hi - lo:.2f}; long delta = {hi - avg:.2f}")


def cli_entrypoint() -> None:
    parser = argparse.ArgumentParser(
        prog="(balance-xdist) show-worker-times",
        description="Show timing data from last tracked test run.",
    )
    parser.add_argument(
        "--plugin-data-dir",
        default=Path.cwd() / ".pytest_cache" / "d" / "balance-xdist",
        type=Path,
        metavar="PATH",
        help="Plugin cache directory.",
    )
    args = parser.parse_args()
    print(f"[balance-xdist] Reading data from {args.plugin_data_dir}")
    show_worker_times(args.plugin_data_dir)
