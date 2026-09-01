"""Custom HTML building blocks for the FraudLens shell.

Each function returns a markup string that is rendered with
``st.markdown(..., unsafe_allow_html=True)``. All dynamic text passes through
:func:`esc` so dataset values (street names, job titles) cannot break the
markup.
"""

from __future__ import annotations

from html import escape

from ..core import config as cfg

STEPS: list[tuple[str, str]] = [
    ("01", "Customer"),
    ("02", "Location"),
    ("03", "Transaction"),
    ("04", "Review"),
    ("05", "Result"),
]

_SHIELD = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#6EA4FF" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2.5 4.5 5.5v6c0 4.6 3.1 8.7 7.5 10 4.4-1.3 7.5-5.4 7.5-10v-6z"/>'
    '<path d="M9.2 12.1l2 2 3.6-3.9"/></svg>'
)

_CHECK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#35A97F" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 6.5 9.6 17 4.5 12"/></svg>'
)

_ALERT = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#E0555C" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3.2 1.9 20.8h20.2z"/><path d="M12 9.3v4.6"/>'
    '<path d="M12 17.4h.01"/></svg>'
)


def esc(value) -> str:
    return escape(str(value), quote=True)


def topbar(model_version: str, feature_count: int) -> str:
    return (
        '<div class="fl-topbar">'
        '<div class="fl-brand">'
        f'<div class="fl-brand__mark">{_SHIELD}</div>'
        '<div>'
        f'<div class="fl-brand__name">{esc(cfg.APP_TITLE)}</div>'
        f'<div class="fl-brand__tag">{esc(cfg.APP_TAGLINE)}</div>'
        '</div></div>'
        '<div class="fl-topbar__meta">'
        '<span class="fl-chip">XGBoost</span>'
        f'<span class="fl-chip">{esc(feature_count)} features</span>'
        f'<span class="fl-chip fl-chip--live">model {esc(model_version)}</span>'
        '</div></div>'
    )


def stepper(current: int) -> str:
    """``current`` is 1-based and matches :data:`STEPS`."""
    items = []
    for index, (num, label) in enumerate(STEPS, start=1):
        if index < current:
            state, glyph = "fl-step--done", "&#10003;"
        elif index == current:
            state, glyph = "fl-step--active", num
        else:
            state, glyph = "", num
        items.append(
            f'<div class="fl-step {state}">'
            f'<div class="fl-step__num">{glyph}</div>'
            f'<div class="fl-step__label">{esc(label)}</div>'
            '</div>'
        )
    return f'<div class="fl-stepper">{"".join(items)}</div>'


def panel_head(eyebrow: str, title: str, description: str) -> str:
    """Step heading.

    The title is a div carrying heading semantics rather than an ``<h1>``:
    Streamlit decorates real headings with its own typography and an anchor
    link, which would override the design system and add a hover icon.
    """
    return (
        '<div class="fl-panel-head">'
        f'<div class="fl-panel-head__eyebrow">{esc(eyebrow)}</div>'
        f'<div class="fl-panel-head__title" role="heading" aria-level="1">'
        f'{esc(title)}</div>'
        f'<p class="fl-panel-head__desc">{esc(description)}</p>'
        '</div>'
    )


def label(text: str, required: bool = True, optional_note: str | None = None) -> str:
    """Field label. Required fields get a red asterisk and nothing else."""
    marker = '<span class="fl-label__req">*</span>' if required else ""
    note = (
        f'<span class="fl-label__opt">{esc(optional_note)}</span>'
        if optional_note else ""
    )
    return f'<div class="fl-label">{esc(text)}{marker}{note}</div>'


def hint(text: str) -> str:
    return f'<p class="fl-hint">{esc(text)}</p>'


def section_title(text: str) -> str:
    return f'<div class="fl-section-title">{esc(text)}</div>'


def divider() -> str:
    return '<div class="fl-divider"></div>'


def note(html_body: str) -> str:
    """Neutral informational block. ``html_body`` is trusted, caller-built."""
    return f'<div class="fl-note">{html_body}</div>'


def readout(value: str | None, unit: str = "", placeholder: str = "—") -> str:
    empty = value in (None, "")
    shown = placeholder if empty else value
    unit_html = f'<span class="fl-readout__unit">{esc(unit)}</span>' if unit and not empty else ""
    cls = "fl-readout fl-readout--empty" if empty else "fl-readout"
    return (
        f'<div class="{cls}">'
        f'<span class="fl-readout__value">{esc(shown)}</span>{unit_html}'
        '</div>'
    )


def errors(messages: list[str], title: str = "Complete these fields to continue") -> str:
    items = "".join(f"<li>{esc(m)}</li>" for m in messages)
    return (
        '<div class="fl-errors">'
        f'<div class="fl-errors__title">{esc(title)}</div>'
        f'<ul>{items}</ul></div>'
    )


def time_separator() -> str:
    return '<div class="fl-time-sep">:</div>'


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------
def review(groups: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    """Render the review summary.

    ``groups`` is ``[(group_title, [(key, value, kind), ...]), ...]`` where
    ``kind`` is ``""``, ``"mono"`` or ``"derived"``.
    """
    blocks = []
    for title, rows in groups:
        cells = []
        for key, value, kind in rows:
            extra = f" fl-review__v--{kind}" if kind else ""
            cells.append(
                '<div>'
                f'<div class="fl-review__k">{esc(key)}</div>'
                f'<div class="fl-review__v{extra}">{esc(value)}</div>'
                '</div>'
            )
        blocks.append(
            '<div class="fl-review__group">'
            f'<div class="fl-review__group-title">{esc(title)}</div>'
            f'<div class="fl-review__grid">{"".join(cells)}</div>'
            '</div>'
        )
    return f'<div class="fl-review">{"".join(blocks)}</div>'


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
def verdict(
    is_fraud: bool,
    probability: float,
    threshold: float,
    metrics: list[tuple[str, str]],
) -> str:
    """Prediction result card.

    Reports only what the model actually produced: the class it assigned and
    the probability behind it. No invented risk bands or tiers.
    """
    state = "fraud" if is_fraud else "safe"
    icon = _ALERT if is_fraud else _CHECK
    status = "Potentially fraudulent" if is_fraud else "No fraud indicated"
    pct = max(0.0, min(1.0, probability)) * 100

    metric_html = "".join(
        '<div class="fl-metric">'
        f'<div class="fl-metric__k">{esc(k)}</div>'
        f'<div class="fl-metric__v">{esc(v)}</div>'
        '</div>'
        for k, v in metrics
    )

    return (
        f'<div class="fl-verdict fl-verdict--{state}">'
        '<div class="fl-verdict__row">'
        f'<div class="fl-verdict__icon">{icon}</div>'
        '<div>'
        '<div class="fl-verdict__label">Model classification</div>'
        f'<div class="fl-verdict__status">{esc(status)}</div>'
        '</div></div>'
        '<div class="fl-prob">'
        '<div class="fl-prob__head">'
        '<span class="fl-prob__name">Fraud probability</span>'
        f'<span class="fl-prob__value">{probability:.2%}</span>'
        '</div>'
        f'<div class="fl-prob__track"><div class="fl-prob__fill" style="width:{pct:.4f}%"></div></div>'
        '<div class="fl-prob__scale"><span>0%</span><span>50%</span><span>100%</span></div>'
        '<div class="fl-prob__threshold">'
        f'The model assigns the fraud class when this probability reaches '
        f'{threshold:.0%}. That is the trained decision rule, not a tuned '
        f'risk band.</div>'
        '</div>'
        f'<div class="fl-metrics">{metric_html}</div>'
        '<div class="fl-disclaimer">'
        '<div class="fl-disclaimer__title">Educational use only</div>'
        'FraudLens is a thesis and portfolio demonstration. It is not a banking '
        'authorisation system and must not be used to make a financial, legal or '
        'account decision. A fraud classification describes the modelled outcome '
        'of this transaction record; it is not a statement about any merchant or '
        'cardholder.'
        '</div>'
        '</div>'
    )


def footer(model_version: str, created: str) -> str:
    return (
        '<div class="fl-footer">'
        f'<span>{esc(cfg.APP_TITLE)} &middot; {esc(cfg.APP_TAGLINE)}</span>'
        f'<span>Model {esc(model_version)}</span>'
        f'<span>Trained {esc(created)}</span>'
        '<span>Inference only &middot; no runtime training</span>'
        '</div>'
    )

