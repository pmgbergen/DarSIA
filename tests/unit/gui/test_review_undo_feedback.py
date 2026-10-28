"""Exercise the actual review HTML and Undo handler in a private browser DOM."""
import ast
import json
from pathlib import Path

import pytest


def test_review_undo_tracks_real_annotations_and_retains_draft_semantics(qtbot):
    """Keep actual editing controls visible while preserving Undo and drafts."""
    core = pytest.importorskip("PySide6.QtWebEngineCore")
    widgets = pytest.importorskip("PySide6.QtWebEngineWidgets")
    from PySide6.QtCore import QUrl
    from shiboken6 import isValid

    source = Path(__file__).resolve().parents[3] / "src/darsia/fluidflower/tools/calibration_pairwise_review.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    template = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "HTML" for target in node.targets))
    bundle = dict(bundle_id="synthetic-review", source_manifest_sha256="1" * 64,
                  runs=[], pairs={}, unpaired_runs=[])
    html = ast.literal_eval(template).replace("__FFAC_PORTABLE_BUNDLE__", json.dumps(bundle)).replace(
        "__FFAC_REVIEW_FLAG_LABELS__", "{}")
    # Reject every external connection; the actual page uses embedded test DATA.
    html = html.replace("<head>", "<head><meta http-equiv=\"Content-Security-Policy\" "
                        "content=\"connect-src 'none'; img-src data:\">")
    profile = core.QWebEngineProfile()
    profile.setHttpCacheType(core.QWebEngineProfile.HttpCacheType.MemoryHttpCache)
    assert profile.isOffTheRecord()
    view = widgets.QWebEngineView()
    page = core.QWebEnginePage(profile, view)
    page.settings().setAttribute(core.QWebEngineSettings.WebAttribute.LocalStorageEnabled, False)
    view.setPage(page)
    qtbot.addWidget(view)

    def evaluate(expression):
        """Wait for a value from the actual page's JavaScript context."""
        values = []
        page.runJavaScript("JSON.stringify(" + expression + ")", values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=10000)
        assert values[0] is not None, expression
        return json.loads(values[0])

    def state():
        """Observe the visible button, real annotation list and memory draft."""
        return evaluate("""({
          caption:byId('undoBox').textContent,
          disabled:byId('undoBox').disabled,
          title:byId('undoBox').title,
          annotations:annotations.length,
          listed:document.querySelectorAll('.annotation-item').length,
          draft:activeSession.drafts.experiment?.annotations.length ?? 0,
          storedDraft:JSON.parse(storageGet(sessionStorageKey('reviewer'))).drafts.experiment?.annotations.length ?? 0
        })""")

    try:
        view.resize(1780, 1000)
        view.show()
        with qtbot.waitSignal(view.loadFinished, timeout=15000) as loaded:
            view.setHtml(html, QUrl("about:blank"))
        assert loaded.args == [True]
        assert evaluate("typeof renderAnnotations === 'function' && state.total === 0 && !STORAGE_PERSISTENT")
        assert evaluate("""(() => {
          const candidate={frame_count:1,elapsed_hours:[0],phase_quiz_proposals:[]};
          PORTABLE_BUNDLE.runs.push('experiment');
          PORTABLE_BUNDLE.pairs.experiment={run:'experiment',candidate_a:candidate,candidate_b:candidate};
          activeSession=newPortableSession('reviewer');
          byId('reviewer').value='reviewer';
          currentRun='experiment';currentPair=portablePair(currentRun);frameCount=1;
          persistPortableSession();state=portableState();renderState();renderAnnotations();
          return reviewerIsReady();
        })()""")
        empty = state()
        assert empty == dict(caption="Undo latest annotation", disabled=True,
                             title="No annotations to undo", annotations=0, listed=0, draft=0, storedDraft=0)
        assert evaluate("""(() => {
          addRectangleAnnotation('A',{x:.1,y:.1},{x:.3,y:.3});
          return annotations.length;
        })()""") == 1
        added = state()
        assert not added["disabled"] and added["title"] == "Remove the most recently added annotation"
        assert added["annotations"] == added["listed"] == added["draft"] == added["storedDraft"] == 1
        for width, height, columns in ((1280, 900, 2), (1780, 1000, 7)):
            view.resize(width, height)
            qtbot.waitUntil(lambda: evaluate("innerWidth") == width, timeout=10000)
            layout = evaluate(r"""(() => {
              const editor=byId('annotationEditor');
              editor.scrollIntoView({block:'center'});
              return {
                active:editor.classList.contains('visible'),
                columns:getComputedStyle(editor).gridTemplateColumns.split(/\s+/).length,
                viewport:document.documentElement.clientWidth,
                document:document.documentElement.scrollWidth,
                height:innerHeight,
                controls:[editor,byId('deleteBox')].map(element=>{
                  const r=element.getBoundingClientRect();
                  return {id:element.id,left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height};
                })
              };
            })()""")
            assert layout["active"] and layout["columns"] == columns, layout
            assert layout["document"] <= layout["viewport"], layout
            for control in layout["controls"]:
                assert control["width"] > 0 and control["height"] > 0, layout
                assert 0 <= control["left"] <= control["right"] <= layout["viewport"] + 1, layout
                assert 0 <= control["top"] <= control["bottom"] <= layout["height"] + 1, layout
        assert evaluate("""(() => {
          byId('undoBox').click();
          return annotations.length;
        })()""") == 0
        assert state() == empty
        assert evaluate("""(() => {
          byId('undoBox').click();
          return Object.keys(activeSession.decisions).length;
        })()""") == 0
        assert state() == empty
    finally:
        view.close()
        page.deleteLater()
        qtbot.waitUntil(lambda: not isValid(page), timeout=10000)
        profile.deleteLater()
        qtbot.waitUntil(lambda: not isValid(profile), timeout=10000)
