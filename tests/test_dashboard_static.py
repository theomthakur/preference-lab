"""Static checks on the dashboard files and the Vercel configuration. They need no browser and no network.
The interactive behaviour (focus, keyboard, mobile widths) was tested separately in a real browser; see docs/ARTIFACTS.md."""
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
HTML = (WEB / "index.html").read_text()
CSS = (WEB / "style.css").read_text()
JS = (WEB / "app.js").read_text()


# ---------- Vercel configuration ----------
def test_vercel_config_serves_the_web_folder_with_safe_headers():
    cfg = json.loads((REPO / "vercel.json").read_text())
    assert cfg["outputDirectory"] == "web" and cfg["framework"] is None
    for name in ("index.html", "style.css", "app.js", "data.json", "favicon.svg"):
        assert (WEB / name).is_file(), name
    headers = {h["key"]: h["value"] for rule in cfg["headers"] if rule["source"] == "/(.*)" for h in rule["headers"]}
    assert headers["X-Content-Type-Options"] == "nosniff"
    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    cache = [h for rule in cfg["headers"] if rule["source"] == "/data.json" for h in rule["headers"]]
    assert cache and "must-revalidate" in cache[0]["value"]


def test_the_strict_csp_is_compatible_with_the_page():
    """The CSP forbids inline styles and scripts, so the page must not contain any."""
    assert not re.search(r"<style\b", HTML) and not re.search(r"<script(?![^>]*\bsrc=)", HTML)
    assert not re.search(r'\sstyle="', HTML), "inline style attribute in index.html"
    assert not re.search(r"\son[a-z]+=", HTML), "inline event handler in index.html"
    assert not re.search(r"""style=['"]""", JS), "app.js builds markup with an inline style attribute, which the CSP would block"
    assert "eval(" not in JS and "new Function" not in JS
    external = set(re.findall(r"""(?:href|src)=["'](https?://[^"']+)""", HTML))
    allowed = ("https://fonts.googleapis.com", "https://fonts.gstatic.com", "https://github.com/theomthakur/preference-lab")
    assert all(u.startswith(allowed) for u in external), external


def test_vercelignore_keeps_weights_and_sources_out_of_a_cli_deploy():
    ignore = (REPO / ".vercelignore").read_text().split()
    for item in (".venv", "runs", "src", "*.safetensors"):
        assert item in ignore, item


# ---------- HTML structure ----------
class Collect(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids, self.refs, self.tags, self.buttons_without_type, self.imgs_without_alt = [], [], [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs); self.tags.append((tag, a))
        if "id" in a: self.ids.append(a["id"])
        for key in ("aria-labelledby", "aria-describedby", "for"):
            if key in a: self.refs += a[key].split()
        if tag == "button" and a.get("type") != "button": self.buttons_without_type.append(a)
        if tag == "img" and "alt" not in a: self.imgs_without_alt.append(a)


def parsed():
    p = Collect(); p.feed(HTML); return p


def test_html_ids_are_unique_and_every_reference_resolves():
    p = parsed()
    assert len(p.ids) == len(set(p.ids)), [i for i in p.ids if p.ids.count(i) > 1]
    runtime_ids = {"case-title", "resp-label", "answer-box", "verdict", "twin"}   # created by app.js when a case is shown
    missing = [r for r in p.refs if r not in p.ids and r not in runtime_ids]
    assert not missing, missing
    assert re.search(r'href="#main"', HTML) and "main" in p.ids                     # skip link target exists


def test_views_nav_and_basic_accessibility_markup():
    p = parsed()
    assert re.search(r'<html lang="en"', HTML) and re.search(r'name="viewport"', HTML)
    views = {a["data-view"] for t, a in p.tags if t == "div" and "data-view" in a}
    nav = {a["data-view-link"] for t, a in p.tags if t == "a" and "data-view-link" in a}
    assert views == nav == {"overview", "system-design", "presentation"}
    assert not p.buttons_without_type and not p.imgs_without_alt
    assert HTML.count("<h1") == 1
    assert 'aria-label="Views"' in HTML and 'id="status"' in HTML and 'role="status"' in HTML
    assert sum(1 for t, a in p.tags if t == "input" and a.get("name") == "wording") == 3
    assert re.search(r"<fieldset[^>]*>\s*<legend>", HTML)


def test_javascript_keeps_focus_and_state_in_attributes():
    assert "aria-current" in JS and "aria-pressed" in JS and "aria-disabled" in JS
    assert "disabled =" not in JS and ".disabled" not in JS, "use aria-disabled so a focused button does not lose focus"
    assert "ArrowRight" in JS and "ArrowLeft" in JS                                # slides respond to arrow keys
    assert "prefers-reduced-motion" in JS                                           # smooth scrolling respects the setting
    # Clicking a model tab must update the answer in place. Re-rendering the whole panel would drop keyboard focus from the tab.
    assert "if (m) { state.model = m.dataset.model; updateModel(true); }" in JS
    update = JS[JS.index("function updateModel"):JS.index("function openCase")]
    assert "renderCase" not in update and "$('case-content').innerHTML" not in update


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_parses():
    r = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_em_dashes_in_the_dashboard_files():
    for name in ("index.html", "app.js", "style.css", "README.md", "generate_data.py"):
        assert "—" not in (WEB / name).read_text(), name


# ---------- CSS: contrast tokens, text size, motion ----------
def tokens():
    root = re.search(r":root\s*\{([^}]*)\}", CSS).group(1)
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", root))


def lum(h):
    h = h.lstrip("#"); r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a, b):
    l1, l2 = lum(a), lum(b); return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


TEXT_PAIRS = [("ink", "paper"), ("ink", "white"), ("text-2", "paper"), ("text-2", "tint"), ("text-2", "soft"), ("muted", "paper"), ("muted", "white"),
              ("muted", "tint"), ("muted", "soft"), ("muted", "warm"), ("amber", "paper"), ("amber", "warm"), ("blue", "white"), ("blue", "soft"),
              ("ev-text", "ev-bg"), ("bad", "bad-bg"), ("good", "good-bg"), ("ondark", "deck"), ("amber-dark", "deck"), ("ondark", "code"),
              ("white", "deck"), ("ink", "warm")]
UI_PAIRS = [("bar-base", "track"), ("bar-sft", "track"), ("bar-dpo-clean", "track"), ("bar-dpo-flipped", "track"), ("bar-grpo", "track"),
            ("line-strong", "white"), ("line-strong", "soft"), ("line-strong", "paper"), ("line-strong", "tint"),
            ("focus", "paper"), ("focus", "white"), ("focus", "tint"), ("focus", "soft"), ("focus-dark", "deck"), ("ink", "white")]


def test_text_colour_pairs_meet_wcag_aa():
    t = tokens()
    bad = {f"{f} on {b}": round(ratio(t[f], t[b]), 2) for f, b in TEXT_PAIRS if ratio(t[f], t[b]) < 4.5}
    assert not bad, bad


def test_graphics_borders_and_focus_rings_meet_3_to_1():
    t = tokens()
    bad = {f"{f} on {b}": round(ratio(t[f], t[b]), 2) for f, b in UI_PAIRS if ratio(t[f], t[b]) < 3.0}
    assert not bad, bad


def test_no_text_smaller_than_12px_and_no_motion():
    sizes = [float(x) for x in re.findall(r"font-size:\s*([\d.]+)px", CSS)] + [float(x) for x in re.findall(r"font:\s*[\d ]*?\s*([\d.]+)px", CSS)]
    assert sizes and min(sizes) >= 12, sorted(set(sizes))[:5]
    assert "scroll-behavior: smooth" not in CSS
    outside_reduced = CSS.split("@media (prefers-reduced-motion: reduce)")[0]
    assert "transition" not in outside_reduced and "animation" not in outside_reduced
    assert "@media (prefers-reduced-motion: reduce)" in CSS


def test_css_has_a_single_palette_and_no_dead_sections():
    assert CSS.count(":root {") == 1, "palette layers should be consolidated into one :root block"
    for dead in (".method-grid", ".limits", ".hero-visual", ".strip"):
        assert dead not in CSS, dead
    assert ":focus-visible" in CSS and ".sr-only" in CSS
