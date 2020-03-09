# Parthenon Development Guide

THIS IS WORK IN PROGRESS AND MEANT AS A BASIS FOR DISCUSSION

## General workflow

### Planning/requesting/announcing changes

If you discover a bug, would like to see a new feature, would like to implement
a new feature, or have any other item/idea that you would like to share
and/or discuss open an issue.

This helps us to keep track of who is working on what and prevent duplicated efforts
and/or allows to pool resources early on.

Use GitHub labels as appropriate and feel free to directly add/tag people to the issue.

### Contributing code

In order to keep the main repository clean, everyone is encouraged to create a 
private fork of the repository.

Afterwards create a feature branch in your private repository and make changes there.

Once all changes are implemented or feedback by other developers is required/helpful
open a pull request again the master branch of the main repository.

In that pull request refer to the issue that you have addressed.

If your branch is not ready for a final merge (e.g., in order to discuss the 
implementation), mark it as "work in progress" by prepending "WIP:" to the subject.

### Merging code

In order for code to be merged into master it must

- obey the style guide (CPPLINT, clang-tidy, ... tbd.)
- pass the existing test suite
- have at least one approval by one member of each physics code
- include tests that cover the new feature (if applicable)
- include documentation (feature or developer; if applicable)



