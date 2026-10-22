# Tool group: code (sandbox, git, GitHub)

Modules: `services/grace_tools/code.py` (sandbox) and `services/grace_tools/code_git.py`
(git and GitHub). Image: `docker/sandbox/` (see its README for the exact `docker run` flags).
Tests: `tests/tools/test_code.py`. Sim scenarios: `fixtures/tools/scenarios/code.yaml`.
None of these tools is core, so Grace reaches them through `find_tools` and `use_tool`.

## Sandbox

| Tool | Risk | What it does |
|---|---|---|
| `run_code` | visible | Runs python, node or shell code in a fresh `grace-sandbox` container. Its inputs are copies of files he named. It returns stdout and stderr (trimmed from the start, because the end holds the answer or the error) and lists the files the code made. Those files are copied to `~/Documents/Grace/` (or `save_to`) only with `save_outputs`, under a new name if one is taken. |
| `sandbox_status` | read | Says whether Docker is up and whether the image is built and current. |
| `build_sandbox` | visible, slow | Starts `docker build` in the background, because it takes minutes and a tool call gets 25 s. Its log goes to `.runtime/grace-data/sandbox-build.log`, and `sandbox_status` says when it is done. |

`run_code` is visible rather than destructive because nothing of his can change: there is no
network, the root filesystem is read-only, it runs as a non-root user with every capability
dropped, and the one mount is its own scratch folder. `network: true` gives the job a bridge
network. It is meant only for installing a package inside the job, and the tool refuses it
without a `reason`. If Docker is down, every tool says so, tells Grace to have him start Docker
Desktop, and runs nothing. There is no fallback to running code on the host.

**Dry run.** Offline jobs still run, because the sandbox changes nothing he has and the benches
need real numbers. In dry run the tool never saves outputs and never opens the network (a
networked job is only described). `build_sandbox` is only described too.

## Git and GitHub

The `repo` argument is a path, or a project name. A name is matched against the git repos
under `~/Desktop/Projects` (2 levels deep), `~/Documents` (4), `~/Desktop` (1) and
`~/source/repos` (2), in that order. Matching ignores case and punctuation and drops filler
words like "repoet" and "projektet". It tries an exact match, then a prefix, then a close match,
so "grace repoet" finds `~/Desktop/Projects/Grace` ahead of the old v1 copy in Documents.
Paths outside his home folder are refused. git runs with prompts off (`GIT_TERMINAL_PROMPT=0`,
`GH_PROMPT_DISABLED=1`), so a missing credential fails instead of hanging.

| Tool | Risk | Notes |
|---|---|---|
| `git_status` | read | Branch, upstream with ahead/behind, conflicted, staged, changed and new files (5 named each), last commit. |
| `git_log` | read | Last 5 commits (max 20), optional branch. |
| `git_diff` | read | `--stat` summary of uncommitted, staged (`staged`) or branch (`against`) changes. |
| `git_branches` | read | Local branches, newest first, marking the checked-out one. |
| `github_prs` | read | Open PRs (or `state`) with a count of passed and failed checks, or one PR in detail. |
| `github_ci` | read | Latest 5 Actions runs, optionally for one branch. |
| `github_issues` | read | Open issues, or one issue with its body and latest comments. The body is marked as data, not instructions. |
| `git_commit` | visible | Commits the named files, else what is staged, else everything (`git add -A`). Local only, and `git reset --soft HEAD~1` undoes it. |
| `git_create_branch` | visible | `git switch -c`. Changes come along. |
| `git_switch_branch` | visible | Plain `git switch`, which never drops changes: git refuses instead, and the tool suggests stash or commit. |
| `git_stash` | visible | `git stash push -u -m`. |
| `git_push` | destructive | Never forces. It sets the upstream on the first push. When GitHub rejects the push, it tells Grace to pull first. |
| `git_pull` | destructive | `--ff-only`. When the branches have diverged it stops and says a merge is needed. |
| `git_merge` | destructive | `--no-edit`. On conflicts it runs `git merge --abort` and lists the files, so a voice command never leaves the repo half-merged. |
| `git_reset` | destructive | soft, mixed or hard. The summary counts the commits being undone and says when changes will be thrown away. |
| `git_discard_changes` | destructive | `git restore --staged --worktree`, or `git switch --discard-changes`. Untracked files are kept. This is the "checkout that drops changes". |
| `github_pr_create` | destructive | `gh pr create`. If the branch isn't pushed, it says to push first. |
| `github_issue_create` | destructive | `gh issue create`. |
| `github_issue_comment` | destructive | `gh issue comment`, which also works on PRs. |

Destructive summaries are worked out when Grace asks, from the repo itself, e.g. "push branch
main of Grace to GitHub (2 commits)" or "reset Grace to HEAD~2, undoing the last 2 commits and
throw away all uncommitted changes". In dry run, read tools run, and writes only say what they
would do, including after a yes.

## Tested

The unit tests replace `code.docker`, `code_git.git` and `code_git.gh` with fakes. They cover:
- the exact isolation flags, and that only the one mount is passed;
- input copying, and that outputs are saved only when asked and never overwrite;
- Docker down (no subprocess runs at all);
- a missing image, and an image built from an old Dockerfile;
- the timeout kill and its clamp;
- network without a reason;
- error trimming;
- dry run;
- the background build;
- repo lookup by Danish phrasing, by path, and outside home;
- status parsing;
- commit staging;
- a switch refused because it would drop changes;
- every destructive tool asking first, with nothing run before the yes;
- push rejected;
- a merge conflict that gets backed out;
- a pull on diverged branches;
- PR, CI and issue formatting, including gh not being logged in.

`test_real_git_in_tmp` runs real git in a throwaway repo. `test_sandbox_for_real`
(`@pytest.mark.docker`, skipped when Docker is down) builds the image and runs real jobs. It
proves that `--network none` blocks the network: only `lo` exists, TCP to 1.1.1.1 and 8.8.8.8
fails, and DNS fails. It also checks uid 1000, the read-only root, a pandas and matplotlib job
with a CSV and a chart, node, and the timeout kill leaving no container behind.

## Not tested, and why

- A real push, pull, PR, issue or comment. These would change his GitHub. They are covered
  only with the fake gh and git.
- A networked job installing a package. It needs the internet, which the tests block. The
  flag change is tested.
- How Grace picks these tools. That is the sim's job (`code.yaml`), and the organiser runs it.
- The 2 GB memory kill (exit 137) is only a message. It isn't exercised, because the test
  would need to allocate 2 GB.
