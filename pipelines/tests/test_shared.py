"""Checks of the code shared by the three pipelines."""

import signal
import sys
import threading

import numpy as np
import pytest

from mantpy_repro.runtime import UNFINISHED, exit_on_sigterm, run_workers
from mantpy_repro.store import Store, claim_mode


def _exits_with(code: int) -> list[str]:
    return [sys.executable, "-c", f"raise SystemExit({code})"]


def test_workers_report_success_an_unfinished_search_or_a_failure():
    assert run_workers([_exits_with(0), _exits_with(0)]) == 0
    assert run_workers([_exits_with(0), _exits_with(UNFINISHED)]) == UNFINISHED
    assert run_workers([_exits_with(UNFINISHED), _exits_with(1)]) == 1
    assert run_workers([_exits_with(0), _exits_with(2)]) == 1


def test_termination_request_becomes_an_exit_inside_the_block():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as caught, exit_on_sigterm():
        signal.raise_signal(signal.SIGTERM)
    assert caught.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before


def test_results_directory_keeps_the_mode_of_its_first_run(tmp_path):
    claim_mode(tmp_path, smoke=True)
    claim_mode(tmp_path, smoke=True)
    assert (tmp_path / "MODE").read_text() == "smoke"
    with pytest.raises(RuntimeError, match="holds smoke results"):
        claim_mode(tmp_path, smoke=False)


def test_a_unit_is_fitted_once_and_a_waiting_process_says_so(tmp_path):
    store, lines, fits = Store(tmp_path), [], []
    holding, release = threading.Event(), threading.Event()

    def slow_fit():
        fits.append(1)
        holding.set()
        release.wait(30)
        return {"value": np.arange(3)}, {}

    def unit():
        return store.unit("unit", request={"seed": 0})

    first = threading.Thread(target=lambda: store.compute(unit(), slow_fit, label="unit", log=lines.append))
    first.start()
    assert holding.wait(30)
    second = threading.Thread(target=lambda: store.compute(unit(), slow_fit, label="unit", log=lines.append))
    second.start()
    for _ in range(300):
        if any(line.startswith("wait") for line in lines):
            break
        threading.Event().wait(0.1)
    release.set()
    first.join(30)
    second.join(30)
    assert len(fits) == 1
    assert sum(line.startswith("wait") for line in lines) == 1
    assert np.array_equal(store.compute(unit(), slow_fit, label="unit")["value"], np.arange(3))


def test_a_damaged_completion_file_is_reported_with_its_folder(tmp_path):
    unit = Store(tmp_path).unit("unit", request={"seed": 0})
    unit.save(0.0, value=np.arange(3))
    unit.marker.write_text("")
    with pytest.raises(RuntimeError, match="cannot be read"):
        unit.done()
