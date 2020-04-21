# Parthenon Development Guide

The following sections cover the current guidelines that all developers are expected
to try to follow.
As with all development these guidelines are not set in stone but subject to discussion
and revision any time as necessary.

## General workflow

### Planning/requesting/announcing changes

If you discover a bug, would like to see a new feature, would like to implement
a new feature, or have any other item/idea that you would like to share
and/or discuss open an issue.

This helps us to keep track of who is working on what and prevent duplicated efforts
and/or allows to pool resources early on.

Use GitHub labels as appropriate and feel free to directly add/tag people to the issue.

### Contributing code

In order to keep the main repository in order, everyone is encouraged to create feature
branches starting with their username, followed by a "/", and ending with a brief
description, e.g., "username/add_feature_xyz".
Working on branches in private forks is also fine but not recommended (as the automated
testing infrastructure will then first work upon opening a pull request).

Once all changes are implemented or feedback by other developers is required/helpful
open a pull request again the master branch of the main repository.

In that pull request refer to the issue that you have addressed.

If your branch is not ready for a final merge (e.g., in order to discuss the 
implementation), mark it as "work in progress" by prepending "WIP:" to the subject.

### Merging code

In order for code to be merged into master it must

- obey the style guide (test with CPPLINT)
- pass the linting test (test with CPPLINT)
- pass the formatting test (see "Formatting Code" below)
- pass the existing test suite
- have at least one approval by one member of each physics/application code
- include tests that cover the new feature (if applicable)
- include documentation in the `doc/` folder (feature or developer; if applicable)

The reviewers are expected to look out for the items above before approving a merge
request.

#### Formatting Code
We use clang-format to automatically format the code. If you have clang-format installed
locally, you can always execute `make format` or `cmake --build . --target format` from
your build directory to automatically format the code.

If you don't have clang-format installed locally, our "Hermes" automation can always
format the code for you. Just create the following comment in your PR, and Hermes will
format the code and automatically commit it:
```
@par-hermes format
```

After Hermes formats your code, remember to run `git pull` to update your local tracking
branch.

If you'd like Hermes to amend the formatting to the latest commit, you can use the
`--amend` option. WARNING: The usual caveats with changing history apply. See:
https://mirrors.edge.kernel.org/pub/software/scm/git/docs/user-manual.html#problems-With-rewriting-history
```
@par-hermes format --amend
```

You will remain the author of the commit, but Hermes will amend the commit for you with
the proper formatting.
Remember - this will change the branch history from your local commit, so you'll need to
run something equivalent to
`git fetch origin && git reset --hard origin/$(git branch --show-current)` to update your
local tracking branch.