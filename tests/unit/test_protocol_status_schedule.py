"""The Protocols step identifies its injection protocol by the schedule, not the file's size and time.

So a corrected injection position keeps the step's confirmation, and a changed
time or rate does not. Synthetic protocol files in a temporary folder.
"""
from darsia.fluidflower import workflow_row_status as status
from darsia.fluidflower.step_confirmations import CONTENT_BASIS, step_basis

from .test_protocol_content import _workbook


def _protocols(tmp_path, *, sheet="ac17", injection=None):
    """The Protocols status of a series whose run uses ``sheet`` of the injection workbook."""
    root = tmp_path / "series"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    imaging = root / "imaging.csv"
    if not imaging.exists():
        imaging.write_text("image,time\nA.ARW,0\n", encoding="utf-8")
    book = injection or root / "injection.xlsx"
    if not book.exists():
        _workbook(book)
    config = dict(data=dict(folders=[str(images)]),
                  protocols=dict(imaging={str(images): str(imaging)}, injection=[str(book), sheet]))
    context = status.WorkflowStatusContext(root, "ac17", config=config)
    return status._configured_protocols(context), book


def _injection(state):
    return next(item for item in state.artifacts if item.path.endswith(("injection.xlsx", "injection.csv")))


def test_the_injection_workbook_enters_by_its_schedule_of_every_worksheet(tmp_path):
    state, book = _protocols(tmp_path)
    item = _injection(state)
    assert state.state == "done" and item.kind == "injection protocol"
    assert item.content.startswith(status.SCHEDULE_FORMAT + ":")
    basis = step_basis(state)
    assert basis.startswith(CONTENT_BASIS)
    # The run loaded does not matter: the workbook holds every run's worksheet.
    assert step_basis(_protocols(tmp_path, sheet="ac18")[0]) == basis
    # A corrected position keeps the basis; the file's bytes and time change.
    _workbook(book, x=0.74)
    corrected = _protocols(tmp_path)[0]
    assert _injection(corrected).content == item.content and step_basis(corrected) == basis
    # A changed start time or rate does not.
    _workbook(book, x=0.74, start="2023-02-09 13:00:05")
    assert step_basis(_protocols(tmp_path)[0]) != basis
    _workbook(book, rate=2e-6)
    assert step_basis(_protocols(tmp_path)[0]) != basis


def test_an_injection_file_the_reader_cannot_parse_keeps_its_size_and_time(tmp_path):
    unreadable = tmp_path / "series" / "injection.csv"
    unreadable.parent.mkdir(parents=True)
    unreadable.write_text("time,pressure\n0,1\n", encoding="utf-8")
    state, _book = _protocols(tmp_path, sheet=None, injection=unreadable)
    item = _injection(state)
    assert item.kind == "saved file" and item.content == ""
    assert not step_basis(state).startswith(CONTENT_BASIS)
