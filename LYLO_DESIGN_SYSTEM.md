# Lylo Landing Page Design System

This file documents the visual rules for the Lylo landing page so new sections use the existing system instead of adding one-off CSS.

## Source of truth

- `static/lylo-design-system.css` — shared landing-page visual system.
- `static/live/lylo-design-system.css` — live wrapper that points to the shared stylesheet.
- `static/et1-intake.css` — specialised ET1 interaction styles only.
- JavaScript files should handle behaviour/content only. They should not inject typography, spacing, button or section CSS.

## Typography

Display font:
- Cormorant Garamond

Body/interface font:
- system sans stack (`-apple-system`, SF Pro, Helvetica Neue, Segoe UI, Roboto, Arial)

Semantic scale:
- `--display-xl` — major section statement
- `--display-lg` — product/demo/mission heading
- `--display-md` — supporting section heading
- `--body-lg` — primary explanatory copy
- `--body-md` — standard body copy
- `--body-sm` — supporting copy
- `--meta` — metadata / labels

## Spacing

Use the shared section rhythm:
- `--section-xl` — major product / trust / conversion section
- `--section-lg` — substantial supporting section
- `--section-md` — bridge / secondary section
- `--section-sm` — compact CTA section

Do not introduce arbitrary section padding unless the component must visually merge with the next section.

## Widths

- `--container: 1280px`
- `--copy: 620px`
- `--copy-wide: 820px`
- `--gutter: clamp(20px, 3vw, 38px)`

## Components

### Major headings
Major page-section headings should use the shared display tiers rather than unique clamp values.

### Body copy
Primary section introductions use `--body-lg`.
Supporting prose should use the body or small-body tier.

### Primary CTA
Hero, demo, middle and final booking CTAs share the same core height, font weight, radius and border treatment.
Header CTA remains intentionally compact.

### Cards
General large cards use `--radius-card`.
Video/media surfaces use `--radius-media`.

## Responsive breakpoint

Primary landing-page breakpoint:
- desktop/tablet split: `979px`

Mobile values are defined through the same tokens instead of separate one-off component scales wherever possible.

## Rule for future edits

Before adding a new numerical value for:
- font size
- section padding
- button height
- border radius
- muted colour

first check whether an existing design token already expresses the intended hierarchy.

Runtime JavaScript must not create style blocks for ordinary landing-page presentation.


## Page stylesheet architecture

Public page CSS now lives in named stylesheets instead of inline HTML:

- `static/lylo-design-system.css` — landing page
- `static/lylo-call.css` — 20-minute call page
- `static/lylo-research.css` — research page
- `static/lylo-about.css` — co-founders/about page
- `static/lylo-privacy.css` — privacy page
- `static/et1-intake.css` — specialised ET1 interaction module

HTML templates should contain structure and content, not large style blocks.
JavaScript should contain behaviour, not layout/theme CSS.
