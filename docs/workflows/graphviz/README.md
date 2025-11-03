# Graphviz Workflow Library

This directory hosts the reorganized Graphviz assets for the 12-Factor Agents plugin bundle. The diagrams were migrated from
`docs/12fa/graphviz` to sit alongside the new `commands/`, `agents/`, and `skills/` top-level directories.

## Structure

- `skills/` – Category-wide skill workflow diagrams and reports
- `agent-mappings/` – Agent topology maps used by swarm coordination playbooks
- `templates/` – DOT templates for creating new diagrams (skill, agent, command)
- `validate-all-diagrams.sh` / `validate-all-diagrams.ps1` – Cross-platform validation helpers
- `master-catalog.json` – Machine-readable catalog of the available diagrams
- `index.html` – Lightweight browser viewer for exploring diagrams

## Usage

Run the validation script that matches your platform to render all DOT files to SVG (Graphviz required):

```bash
cd docs/workflows/graphviz
bash validate-all-diagrams.sh
```

On Windows PowerShell:

```powershell
cd docs/workflows/graphviz
./validate-all-diagrams.ps1
```

Open `index.html` in a browser to browse the generated SVGs. The viewer lists every DOT file under `skills/` and
`agent-mappings/` and links to the rendered diagrams when present.
