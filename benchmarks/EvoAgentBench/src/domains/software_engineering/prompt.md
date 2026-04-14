WRAPPER_PATH: {wrapper_path}

You need to fix a bug in the {repo} repository. Time limit: {timeout_min} minutes.

[STRICT RULES]
- All commands MUST be executed via WRAPPER_PATH
- NEVER use host file tools (read, write, edit, etc.) - they operate on the host, not the task environment
- NEVER use exec directly to run commands (must go through WRAPPER_PATH)
- To write files, use WRAPPER_PATH write, NOT tmux-run + cat/heredoc (escaping issues)

How to operate:
  Run command: exec("{wrapper_path} tmux-run \\"command\\" wait_seconds")
  Poll/wait:   exec("{wrapper_path} tmux-run \\"\\" 10")
  Write file:  exec("{wrapper_path} write /target/path << 'EOF'\nfile content\nEOF")
  Interrupt:   exec("{wrapper_path} tmux-run \\"ctrl-c\\" 3")

Notes:
  - If you see [STILL RUNNING: xxx], the previous command is still executing. You can:
    a) Keep waiting: exec("{wrapper_path} tmux-run \\"\\" 30") (empty command + longer wait)
    b) Interrupt: exec("{wrapper_path} tmux-run \\"ctrl-c\\" 3")
    c) Do NOT send new commands while one is still running

## Workflow
1. Understand the bug from the problem statement
2. Locate the relevant source files
3. Reproduce the bug (write a small test if helpful)
4. Fix the bug
5. Verify your fix works

## Bug Description

{problem}

Reply TASK_COMPLETE when done.
