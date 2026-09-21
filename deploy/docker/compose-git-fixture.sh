#!/bin/sh
set -eu

source=/fixture/source
repository=/fixture/repo/fixture-vault.git

# `docker compose run smoke` starts dependency services when they have exited.
# Preserve the already-served repository so that this lifecycle action cannot
# change the commit workers are reconciling.
if [ -f "$repository/HEAD" ]; then
    exit 0
fi

worktree=$(mktemp -d)
trap 'rm -rf "$worktree"' EXIT

mkdir -p "$repository"
cp -R "$source"/. "$worktree"/
git -C "$worktree" init --quiet
git -C "$worktree" config user.name fixture
git -C "$worktree" config user.email fixture@example.invalid
git -C "$worktree" add --all
GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
    git -C "$worktree" commit --quiet --no-gpg-sign --message fixture
git -C "$worktree" branch -M main
git clone --quiet --bare "$worktree" "$repository"
git --git-dir="$repository" update-server-info
