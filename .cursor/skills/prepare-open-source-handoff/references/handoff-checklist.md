# Open-Source Handoff Checklist

Use this checklist at the end of development before publishing or transferring maintainership.

## 1. Repository Hygiene

- [ ] Remove generated caches and transient artifacts
- [ ] Remove local machine metadata files (`.DS_Store`, `Thumbs.db`, etc.)
- [ ] Confirm no credentials or `.env` secrets are committed
- [ ] Confirm `.gitignore` covers expected local/build outputs
- [ ] Keep repository root concise and purpose-driven

## 2. Documentation Quality

- [ ] README explains project value, scope, and use cases
- [ ] README includes installation, usage, and development instructions
- [ ] README includes contribution pathway for external collaborators
- [ ] README links to `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` when present
- [ ] Commands in README are copy-paste runnable

## 3. Contributor Experience

- [ ] Contribution guidelines define branch/PR expectations
- [ ] Issue and PR templates exist or rationale for omission is documented
- [ ] Testing and linting entry points are documented
- [ ] Project structure and key directories are explained
- [ ] Expected support channels are documented

## 4. Governance and Risk

- [ ] License is valid and consistent with README
- [ ] Security reporting process is documented
- [ ] Maintainer ownership and response expectations are clear
- [ ] Breaking change policy or versioning approach is documented

## 5. Release Readiness

- [ ] CI checks pass on default branch
- [ ] Version/tag/release notes process is documented
- [ ] Changelog strategy is clear (if applicable)
- [ ] Optional: add a "Roadmap" section to attract contributors
