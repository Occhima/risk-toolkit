# containers

Central place for the project's container images.

- `Dockerfile` — base development image (Python 3.12 + uv + system tools).
  Referenced by `.devcontainer/devcontainer.json` via `../containers/Dockerfile`
  with the build context set to the repository root.

When the CLI lands it gets its own image here (e.g. `cli.Dockerfile`) that
builds on the same workspace, so all container definitions stay in one folder.
