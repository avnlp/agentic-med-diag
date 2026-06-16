### Related Issues

- fixes #issue-number

### Proposed Changes:

 <!--- In case of a bug: Describe what caused the issue and how you solved it -->
 <!--- In case of a feature: Describe what did you add and how it works -->

### How did you test it?

<!-- unit tests, integration tests, manual verification, instructions for manual tests -->

### Notes for the reviewer

<!-- E.g. point out section where the reviewer  -->

### Checklist

- [ ] Code follows project style (`make lint-check`)
- [ ] Type annotations pass (`make lint-typing`)
- [ ] No typos (`make lint-typos`)
- [ ] No new security issues (`make security-bandit`)
- [ ] Dependency vulnerabilities checked (`make security-audit`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] I have updated the related issue with new insights and changes
- [ ] I added unit tests and updated the docstrings
- [ ] I've used one of the [conventional commit types](https://www.conventionalcommits.org/en/v1.0.0/) for my PR title: `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`.
- [ ] I documented my code
