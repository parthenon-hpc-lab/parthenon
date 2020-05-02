#!/usr/bin/env bash

# This script forces CI to re-run by churning the commits that this branch
# points to on the remote. This method doesn't involve re-writing history, which
# is good, but it's possible there's another way without having to change to an
# older commit first.

# exit when any command fails
set -e

# Make sure the branch is up-to-date to avoid rolling-back any changes made in the
# meantime.
git pull
# We could just store this in a local variable, but this risk is losing track of
# the newer commit if something goes wrong.
git tag retrigger-ci-tmp
# Reset to the previous commit
git reset --hard HEAD^
# Force push that fallback
git push --force
# Reset to the latest commit.
git merge --ff-only retrigger-ci-tmp
# Doesn't require force - we're pushing a newer commit
git push
# Remove the tag - we don't need it anymore
git tag -d retrigger-ci-tmp