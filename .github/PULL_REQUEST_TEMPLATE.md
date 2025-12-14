## Description

Brief description of what this PR does.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] New plugin

## Related Issues

Closes #(issue number)

## Changes Made

- Change 1
- Change 2
- Change 3

## How to Test

1. Step 1
2. Step 2
3. Step 3

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have run `black` and `isort` on my code
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass (`pytest`)
- [ ] I have run the architecture check (`python -m tools.contract_map --fail-on-errors`)
- [ ] I have updated documentation as needed
- [ ] My changes don't introduce new warnings

## Architecture Compliance

- [ ] No imports from `rag/` or `ingest/` in `core/`
- [ ] No imports from `rag/` in `ingest/`
- [ ] New plugins follow the Protocol pattern
- [ ] Config-driven design (no hardcoded provider selection)

## Screenshots (if applicable)

## Additional Notes

Any additional information reviewers should know.
