# Package Development Workflows

This document describes workflows for developing the Skynet autonomous learning system, built in Sindarin.

This package tracks active ML / RL research. The workflow rules below are stricter than a typical package: every algorithmic suggestion must be grounded in current state-of-the-art literature, not in memory or in the existing code.

See ./.sn/sindarin-pkg-agents/README.md for Sindarin language and tooling reference.

## !!! ALWAYS DO THIS !!!

!!! ALWAYS ASK FOR PERMISSION BEFORE MAKING CODE CHANGES !!!
!!! CHANGING CODE WITHOUT MY APPROVAL IS A VIOLATION OF THIS INSTRUCTION !!!
!!! NEVER MENTION A FAILURE/ISSUE AS "PRE-EXISTING", INVESTIGATE THE ISSUE AND SUGGEST A FIX !!!

!!! WHEN I SAY "ALWAYS ASK FOR PERMISSION BEFORE MAKING CODE CHANGES" I MEAN:
- DO NOT write code, edit files, or run edit/write tools until the user explicitly says "yes" or "go ahead"
- DO NOT propose a plan and then execute it in the same message
- DO NOT revert, fix, or "clean up" code without explicit approval
- "Can I proceed?" is ASKING. Touching a file is NOT asking.
- INVESTIGATE and REPORT findings. Then WAIT for instruction.
- The ONLY exception is reading files for research purposes !!!

## !!! WORKFLOW ENFORCEMENT !!!

The correct workflow is ALWAYS:
1. RESEARCH the problem (read files, grep, explore)
2. REPORT your findings to the user
3. SUGGEST what you think should change (describe in words, do NOT write the code yet)
4. WAIT for the user to approve or redirect
5. ONLY THEN make the approved changes — nothing more, nothing less
6. If you discover additional issues during implementation, STOP and report them. DO NOT fix them without approval.

NEVER skip steps 3 and 4. NEVER combine steps 3-5 into one action.

## !!! RESEARCH-FIRST FOR ML/RL !!!

Before suggesting *any* algorithmic change in this package — especially for reinforcement learning (PPO, GRPO, advantage estimation, value functions, optimizers, loss functions, exploration, normalization, KL handling, reward shaping, etc.) — you MUST perform online research to confirm current state-of-the-art.

- Use WebSearch / WebFetch against recent arXiv papers, reference implementations (TRL, CleanRL, OpenAI Spinning Up, DeepMind), and recognized practitioner blogs.
- Cite the specific sources (paper title + year, repo + file, blog post URL) in your report. "I think this is correct" is NOT sufficient. "Standard practice is..." without a citation is NOT sufficient.
- Memory and training-data recall are NOT acceptable substitutes for live research. The field moves fast and your priors are stale.
- If you cannot access the web for any reason, say so explicitly and STOP. Do not fall back to memory and pretend it is research.
- If research surfaces a conflict between the current code and SOTA, treat it as an audit finding: report it and WAIT for instruction. Do not silently fix it (per the existing permission rule).

This applies to suggestions, not just to writing code. Reporting "we should add X" without having researched X first is a workflow violation.

## !!! CONTINUOUS SOTA AUDITING !!!

Whenever you read RL / training / optimizer / loss / advantage / normalization code for any reason — even unrelated to the user's immediate request — audit it against current SOTA in passing.

- Surface any drift, theoretical bugs, outdated formulations, sign errors, known-bad patterns, or numerical issues as findings in your report.
- Theoretical correctness and up-to-dateness outrank consistency with the existing implementation. The code in this repo is NOT authoritative — the literature is.
- Do not assume an existing pattern is correct just because it is committed. Do not assume a recent commit is correct just because it is recent. Verify against sources.
- Audit findings are reports, not authorizations. Report them and wait, same as any other suggestion.

## API DESIGN PRINCIPLES

- **Breaking changes are fine.** This package has no stability contract. Do not preserve old APIs, do not add compatibility shims, do not deprecate-then-remove. If a cleaner design exists, propose replacing the old one outright.
- **Clean, composable API.** Prefer small orthogonal pieces (strategies, optimizers, schedulers, normalizers, advantage estimators) that consumers compose, over monolithic configs with dozens of flags.
- **Consumer choice over opinionated defaults.** Where SOTA offers multiple valid approaches (e.g. advantage normalization on/off, KL penalty vs. clip, GAE vs. Monte Carlo returns, AdamW vs. Lion, value clipping on/off), expose them as selectable strategies rather than hard-coding one. Sensible defaults are fine, but every choice must be overridable.
- **No speculative knobs.** Only add a configuration option when SOTA actually presents a real choice between defensible alternatives. Do not add flags "just in case" or for hypothetical future needs.

