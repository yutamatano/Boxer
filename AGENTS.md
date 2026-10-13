# Repository workflow

- The user edits and runs this repository on the lab PC through VS Code Remote SSH. Do not assume a desktop display is available on the remote host. Provide downloadable artifacts or localhost services accessible through VS Code port forwarding.
- After editing files, run appropriate validation, commit the task's changes, and push to the configured GitHub remote. This is the user's standing request, including future tasks. Do not commit recordings, model weights, generated outputs, credentials, or unrelated changes. If pushing is blocked, report the exact blocker and the unpushed commit.
- Keep source frame IDs and integer nanosecond timestamps intact. Never turn failed or nonmetric camera tracking into valid metric poses by relaxing checks or filling gaps.
