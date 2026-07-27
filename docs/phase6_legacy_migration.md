# Phase 6 legacy migration

Phase 6 removed the Model123 and time-scaled/generator-defect research from the
active Paper 1 source tree, import graph, tests, documentation, wheel, and
sdist.

The deletion-before tree is anchored by annotated Git tag
`pre-phase6-model123`, which points to commit
`9320d1e86c7a2212f56cf4c64a326727cdd03b43`.

Restore the complete archived tree without changing the current branch:

```bash
git worktree add ../POL_for_PDE-model123-archive pre-phase6-model123
```

Restore one file for inspection:

```bash
git show pre-phase6-model123:pol/model123_1d/experiments.py
```

The neutral numerical implementations used by active Paper 1 are now
`pol/numerics/burgers.py`, `pol/numerics/etdrk4.py`, and
`pol/numerics/initial_conditions.py`. Their Phase 6 migration was an exact
move before active imports were switched.
