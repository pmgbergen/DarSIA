"""Synthetic Qt metadata-only controls; no actual OS probe, job or store map."""
import hashlib
import threading
from types import SimpleNamespace

from darsia.fluidflower import store_map_ui as ui
from darsia.fluidflower.store_map import parse_store_map


RAW = b'\xef\xbb\xbf[machine]\r\nid="Main"\r\n[stores.S]\r\nseries="/data/series"\r\nresults="/products"\r\n[future]\r\nkeep=17\r\n'


def reader_for(path, calls):
    def read(selected, **kwargs):
        calls.append((str(selected), threading.get_ident(), kwargs))
        return parse_store_map(path.read_bytes(), source={"path": str(path)})
    return read


def test_construct_context_and_settings_open_never_load_or_probe(qtbot, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Passive navigation must not read/probe/write")
    panel = ui.StoreAvailabilityPanel(path=tmp_path/"stores.toml", reader=forbidden, availability=forbidden)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path/"S", "S")
    panel.set_context(tmp_path/"T", "T")
    assert panel.report is None and panel.check_button.isEnabled()
    panel.open_settings()
    qtbot.addWidget(panel._dialog)
    assert panel._dialog._thread is None and not panel._dialog.save_button.isEnabled()
    panel._dialog.reject()
    assert panel.can_close()


def test_close_discards_blocked_availability_but_keeps_settings_guard(qtbot, tmp_path):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def read(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        returned.set()
        return SimpleNamespace(machine_id="Main")
    panel = ui.StoreAvailabilityPanel(path=tmp_path / "stores.toml", reader=read,
        availability=lambda *args: dict(roots=[]))
    qtbot.addWidget(panel)
    panel.set_context(tmp_path / "S", "S")
    panel.check()
    qtbot.waitUntil(entered.is_set)
    assert not panel.can_close()
    panel.prepare_close()
    assert panel.can_close() and not release.is_set()
    panel.open_settings()
    assert not panel.can_close()
    panel._dialog.reject()
    release.set()
    qtbot.waitUntil(returned.is_set)
    assert panel.report is None and panel.can_close()


def test_explicit_check_offloads_only_directory_metadata_and_context_switch_drops_it(qtbot, tmp_path):
    path=tmp_path/"stores.toml"; path.write_bytes(RAW)
    entered=threading.Event(); release=threading.Event(); calls=[]
    def availability(stores, series):
        entered.set(); release.wait(5)
        return {"series_id": series, "machine_id": stores.machine_id,
                "roots":[{"alias":"raw", "path":"/raw", "available":False, "reason":"missing"}]}
    panel=ui.StoreAvailabilityPanel(path=path, reader=reader_for(path,calls), availability=availability)
    qtbot.addWidget(panel); panel.set_context(tmp_path/"S", "S")
    try:
        panel.check()
        qtbot.waitUntil(entered.is_set)
        assert not panel.can_close()
        panel.set_context(tmp_path/"T", "T")
    finally:
        release.set()
    qtbot.waitUntil(lambda:panel._thread is None)
    assert calls[0][1] != threading.get_ident()
    assert panel.report is None and "T" in panel.summary.text()
    assert panel.can_close()


def test_explicit_missing_report_never_claims_input_hashes_or_worker_eligibility(qtbot, tmp_path):
    path=tmp_path/"stores.toml"; path.write_bytes(RAW)
    calls=[]
    def availability(stores, series):
        return {"series_id":series,"roots":[{"alias":"raw","path":"/raw", "available":False,"reason":"missing"}]}
    panel=ui.StoreAvailabilityPanel(path=path,reader=reader_for(path,calls),availability=availability)
    qtbot.addWidget(panel); panel.set_context(tmp_path/"S","S"); panel.check()
    qtbot.waitUntil(lambda:panel._thread is None)
    assert "unavailable" in panel.summary.text()
    assert "not verified" in panel.details.toPlainText()
    assert "No job was started" in panel.details.toPlainText()


def test_settings_unchanged_save_preserves_original_bom_crlf_and_unknown_fields(qtbot,tmp_path):
    path=tmp_path/"stores.toml"; path.write_bytes(RAW)
    calls=[]; writes=[]
    def writer(path,raw,*,expected_bytes,**kwargs):
        writes.append((path,raw,expected_bytes))
        return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"previous_copy":"fixture-previous"}
    dialog=ui.StoreSettingsDialog(path=path,reader=reader_for(path,calls),writer=writer)
    qtbot.addWidget(dialog)
    assert not calls and not writes
    dialog.load(); qtbot.waitUntil(lambda:dialog._thread is None)
    assert dialog.editor.toPlainText()==dialog._displayed_text
    dialog.save(); qtbot.waitUntil(lambda:dialog._thread is None)
    assert writes[0][1:] == (RAW,RAW)
    assert path.read_bytes()==RAW  # fake writer was the only save callback


def test_settings_reject_does_not_destroy_running_metadata_thread(qtbot,tmp_path):
    path=tmp_path/"stores.toml"; path.write_bytes(RAW)
    entered=threading.Event(); release=threading.Event()
    def reader(*args,**kwargs):
        entered.set(); release.wait(5)
        return parse_store_map(RAW,source={"path":str(path)})
    dialog=ui.StoreSettingsDialog(path=path,reader=reader)
    qtbot.addWidget(dialog); dialog.show()
    try:
        dialog.load(); qtbot.waitUntil(entered.is_set)
        dialog.reject()
        assert dialog.isVisible() and dialog._thread is not None
    finally:
        release.set()
    qtbot.waitUntil(lambda:dialog._thread is None)
    dialog.reject()
