# Contributing to Flympics

Flympics is a teaching project. Fork the repo, work on a branch, and open a pull request.

## Workflow

1. Fork the repository and clone your fork.
2. Set up the environment (see the README for macOS, Linux and Windows).
3. Create a branch: `git checkout -b my-change`.
4. Make a small, focused change and run `python -m pytest -q`.
5. Push to your fork and open a pull request.

Pick a task from the issues labeled `good first issue`, `intermediate` or `advanced`.

## Ground rules

- **Be honest about scope.** This is a small model of a 135-neuron MANC circuit. It is not a whole-fly-brain or biological-behavior simulation. In every change, say what comes from data and what is a modeled assumption.
- **No hardcoded winning actions and no falsified scores.** Improve performance only through explicit training. A poor result is a valid result.
- **Keep events independent of the neural code.** Events expose observations and accept legal actions; controllers consume them.
- **Preserve determinism.** Keep the fixed 0.1 s step and seeded runs.
- **Keep the circuit as is.** Do not densify connectivity, flip the `[post, pre]` matrix orientation, or change normalization silently.
- **Stay offline and light.** No raw data downloads in normal runs, and no React, Node tooling, or whole-brain datasets.
- **Do not hand-edit circuit CSVs.** Change the extraction script and provenance together, and keep the CC BY 4.0 attribution.

## Experiments

Write down your hypothesis and your metric before you run anything, and report results across several seeds.
