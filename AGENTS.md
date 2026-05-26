# AGENTS.md

Revit 2026 add-in (.NET 8) that implements the Venezuelan COVENIN E4 construction-norm coding inside Autodesk Revit. Code is minimal so far; most of the value is in `Ponchevit/docs/`, which is the design spec.

## Layout

- `Ponchevit.slnx` — solution (new XML format) at repo root.
- `Ponchevit/Ponchevit.csproj` — the only buildable project.
- `Ponchevit/Ponchevit.cs` — `IExternalApplication` entrypoint (`Ponchevit.App`). Creates ribbon tab "Ponchevit USM", panel "Acciones".
- `Ponchevit/ButtonOne.cs` — example `IExternalCommand` (`Ponchevit.CommandOne`).
- `Ponchevit/manifest/Ponchevit.addin` — Revit add-in manifest. `AddInId` GUID `1bfbb086-06af-4ddf-b84b-99d3fad0366f` must stay stable.
- `Ponchevit/docs/` — authoritative design spec (read before changing UI/data behavior):
  - `plugin.md` — three planned commands: `Agregar Familia`, `Asignar Código`, `Reconocer Elemento`. Defines panel layouts and execution logic.
  - `partidas.md` — hierarchical code structure (Capítulo / Subcapítulo / Sección), variable digit length, prefix-mask queries.
  - `tablas.md` — SQLite DAG schema: `Covenin_Columnas`, `Covenin_Valores` (Flyweight), `Covenin_Conexiones` (adjacency via `Parent_Id`, with `Codigo_Aportado` concatenated left-to-right; "empty bridges" inherit `Parent_Id`; final codes capped at 10 digits).
- `Partidas/`, `Tablas/` (repo root, outside csproj) — Python data-extraction/scraping pipelines that produce the eventual SQLite content. Not built by the solution; treat as separate tooling.
- `.opencode/agent/` — architect / coder / docu_writer subagent definitions.

## Build / run

- Windows-only build. `Ponchevit.csproj` references `RevitAPI.dll` / `RevitAPIUI.dll` via hardcoded `HintPath` to `C:\Program Files\Autodesk\Revit 2026\`. Revit 2026 must be installed at that path or the references must be repointed.
- `dotnet build Ponchevit.slnx` (or build `Ponchevit/Ponchevit.csproj`). Target framework `net8.0`, nullable enabled.
- PostBuild step copies the built DLL **and** `manifest/Ponchevit.addin` to `%AppData%\Autodesk\Revit\Addins\2026\`. Uses `cmd` `copy /Y`, so it only works on Windows (not WSL/Linux dotnet).
- The `<Assembly>` element inside `Ponchevit.addin` is an absolute path containing a specific Windows username — adjust per machine if needed; do not commit machine-specific edits unless asked.
- No tests, no lint config, no CI. Manual verification is loading the add-in in Revit 2026 and clicking the "Command One" button on the "Ponchevit USM" tab.

## Adding a ribbon command

1. New class implementing `IExternalCommand` with `[Transaction(TransactionMode.ReadOnly|Manual)]`.
2. Register it in `Ponchevit.App.OnStartup` via `PushButtonData(name, label, assemblyPath, fullyQualifiedClassName)` and `panel.AddItem(...)`. The existing `CommandOne` registration is the template.
3. Transactions that modify the model must use `TransactionMode.Manual` and an explicit `Transaction` scope — `CommandOne` is `ReadOnly` and is not a template for write operations.

## Conventions

- UI strings and domain terms are Spanish (`Capítulo`, `Subcapítulo`, `Sección`, `Partida`, `Agregar`, `Asignar`, `Reconocer`). Keep them in Spanish; match `docs/` terminology exactly.
- Trust `docs/` over any speculation about the data model. If docs and code diverge, the docs describe the intended target state — flag the gap rather than silently re-deriving it.
- Code base is currently tiny; do not introduce frameworks, DI containers, or large abstractions without an explicit request.
