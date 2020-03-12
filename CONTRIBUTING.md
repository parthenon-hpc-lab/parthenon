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
- pass the existing test suite
- have at least one approval by one member of each physics/application code
- include tests that cover the new feature (if applicable)
- include documentation in the `doc/` folder (feature or developer; if applicable)

The reviewers are expected to look out for the items above before approving a merge
request.


