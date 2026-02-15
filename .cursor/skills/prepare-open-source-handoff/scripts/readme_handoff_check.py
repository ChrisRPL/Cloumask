#!/usr/bin/env python3
"""Audit README.md for open-source handoff readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)

REQUIRED_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Overview", ("overview", "about", "introduction")),
    ("Features", ("features", "capabilities", "highlights")),
    ("Installation", ("installation", "install", "setup")),
    ("Usage", ("usage", "examples", "how to use")),
    ("Development", ("development", "local development", "developer setup")),
    ("Contributing", ("contributing", "contribution")),
    ("Code of Conduct", ("code of conduct",)),
    ("Security", ("security", "security policy", "reporting security issues")),
    ("License", ("license",)),
]

RECOMMENDED_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Prerequisites", ("prerequisites", "requirements")),
    ("Configuration", ("configuration", "environment variables", "config")),
    ("Testing", ("testing", "tests", "quality checks")),
    ("Roadmap", ("roadmap", "future work", "planned work")),
    ("Support", ("support", "community", "maintainers", "contact")),
]

SECTION_STUBS: dict[str, str] = {
    "Overview": "## Overview\nDescribe what the project does, who it is for, and why it exists.\n",
    "Features": "## Features\n- List core capabilities\n- Highlight unique strengths\n",
    "Installation": "## Installation\n```bash\n# Add install steps\n```\n",
    "Usage": "## Usage\n```bash\n# Add usage examples\n```\n",
    "Development": "## Development\nDocument local setup, coding workflow, and quality checks.\n",
    "Contributing": "## Contributing\nExplain contribution flow and link to `CONTRIBUTING.md` if available.\n",
    "Code of Conduct": "## Code of Conduct\nState expected behavior and link to `CODE_OF_CONDUCT.md`.\n",
    "Security": "## Security\nExplain how to report vulnerabilities and link to `SECURITY.md`.\n",
    "License": "## License\nState the license and reference the `LICENSE` file.\n",
    "Prerequisites": "## Prerequisites\nList required tools, runtime versions, and system dependencies.\n",
    "Configuration": "## Configuration\nDocument required environment variables and configuration files.\n",
    "Testing": "## Testing\n```bash\n# Add test commands\n```\n",
    "Roadmap": "## Roadmap\nOutline near-term priorities and planned improvements.\n",
    "Support": "## Support\nDescribe where contributors/users can ask questions and get help.\n",
}


def normalize_heading(text: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", text)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value.lower())
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_normalized_headings(markdown: str) -> list[str]:
    return [normalize_heading(match.group(1)) for match in HEADING_RE.finditer(markdown)]


def section_present(headings: list[str], aliases: tuple[str, ...]) -> bool:
    for heading in headings:
        for alias in aliases:
            if alias == heading or alias in heading:
                return True
    return False


def missing_sections(
    headings: list[str], section_spec: list[tuple[str, tuple[str, ...]]]
) -> list[str]:
    missing: list[str] = []
    for title, aliases in section_spec:
        if not section_present(headings, aliases):
            missing.append(title)
    return missing


def append_stubs(
    readme_path: Path, missing_required: list[str], missing_recommended: list[str], include_recommended: bool
) -> list[str]:
    sections = list(missing_required)
    if include_recommended:
        sections.extend(missing_recommended)

    if not sections:
        return []

    existing = readme_path.read_text(encoding="utf-8")
    chunks = [f"\n\n## {title}\nTODO: add project-specific details.\n" for title in sections]
    for idx, title in enumerate(sections):
        stub = SECTION_STUBS.get(title)
        if stub:
            chunks[idx] = f"\n\n{stub.rstrip()}\n"

    readme_path.write_text(existing.rstrip() + "".join(chunks) + "\n", encoding="utf-8")
    return sections


def write_report(
    output_path: Path,
    readme_path: Path,
    required_missing: list[str],
    recommended_missing: list[str],
    appended: list[str],
) -> None:
    lines = [
        "# README Open-Source Handoff Audit",
        "",
        f"- README: `{readme_path}`",
        f"- Missing required sections: **{len(required_missing)}**",
        f"- Missing recommended sections: **{len(recommended_missing)}**",
        "",
        "## Missing Required",
    ]
    if required_missing:
        lines.extend(f"- {item}" for item in required_missing)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Recommended"])
    if recommended_missing:
        lines.extend(f"- {item}" for item in recommended_missing)
    else:
        lines.append("- None")

    lines.extend(["", "## Appended Sections"])
    if appended:
        lines.extend(f"- {item}" for item in appended)
    else:
        lines.append("- None")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit README for open-source handoff readiness.")
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument(
        "--append-missing",
        action="store_true",
        help="Append stub sections for missing required sections",
    )
    parser.add_argument(
        "--append-recommended",
        action="store_true",
        help="When appending, include missing recommended sections too",
    )
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if required sections are missing")
    parser.add_argument("--report", type=Path, help="Write markdown audit report")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readme_path = Path(args.readme).resolve()
    if not readme_path.exists():
        print(f"error: README not found: {readme_path}", file=sys.stderr)
        return 2

    body = readme_path.read_text(encoding="utf-8")
    headings = extract_normalized_headings(body)

    required_missing = missing_sections(headings, REQUIRED_SECTIONS)
    recommended_missing = missing_sections(headings, RECOMMENDED_SECTIONS)

    appended: list[str] = []
    if args.append_missing:
        appended = append_stubs(
            readme_path=readme_path,
            missing_required=required_missing,
            missing_recommended=recommended_missing,
            include_recommended=args.append_recommended,
        )
        body = readme_path.read_text(encoding="utf-8")
        headings = extract_normalized_headings(body)
        required_missing = missing_sections(headings, REQUIRED_SECTIONS)
        recommended_missing = missing_sections(headings, RECOMMENDED_SECTIONS)

    if args.report:
        write_report(
            output_path=args.report.resolve(),
            readme_path=readme_path,
            required_missing=required_missing,
            recommended_missing=recommended_missing,
            appended=appended,
        )

    if args.json:
        payload = {
            "readme": str(readme_path),
            "missing_required": required_missing,
            "missing_recommended": recommended_missing,
            "appended_sections": appended,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"README: {readme_path}")
        print(f"Missing required sections: {len(required_missing)}")
        for title in required_missing:
            print(f"- {title}")
        print(f"\nMissing recommended sections: {len(recommended_missing)}")
        for title in recommended_missing:
            print(f"- {title}")
        if appended:
            print(f"\nAppended sections: {len(appended)}")
            for title in appended:
                print(f"- {title}")

    if args.strict and required_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
