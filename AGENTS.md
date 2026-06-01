# AGENTS.md

Revit 2026 add-in (.NET 8) that implements the Venezuelan COVENIN E4 construction-norm coding inside Autodesk Revit. The plugin's value sits in four coordinated workflows — *Mapeo de Materiales* (per-project mapping from Revit materials to COVENIN material categories), *Agregar Familia* (parametric generation of code-compliant elements), *Asignar Código* (manual + auto-prefill coding of existing elements), and the *Codificación Dashboard* (project-wide audit + Revit-native `ViewSchedule` report).

## Working with this repo

- Start every session by reading `Ponchevit/docs/roadmap.md` first to see the current phase and what is next.
- Update `Ponchevit/docs/roadmap.md` on completion with the checked checkbox + commit SHA.
- If you deviate from the plan, append a same-day entry to `Ponchevit/docs/decisions.md` (ADR format, newest first).
- Trust `docs/` over speculation about the data model. If docs and code diverge, the docs describe the intended target state — flag the gap rather than silently re-deriving it.
- Documentation is in English; domain terms stay in Spanish (`Capítulo`, `Subcapítulo`, `Sección`, `Partida`, `Agregar`, `Asignar`, `Muro`, `Codificación`, `Mapeo de Materiales`, etc.).
- **Do not run `git commit` / `git push` / `git add` unprompted** — the repo owner runs all git write operations.

## Layout

- `Ponchevit.slnx` — solution (new XML format) at repo root.
- `Ponchevit/Ponchevit.csproj` — the only buildable project (net8.0, nullable enabled). References `CommunityToolkit.Mvvm` (MVVM source generators) and `Microsoft.Data.Sqlite`.
- `Ponchevit.Tests/Ponchevit.Tests.csproj` — xUnit (net8.0); covers Domain + Data layers (which are pure C#, no RevitAPI refs).
- `Ponchevit/App.cs` — `IExternalApplication` entrypoint, creates ribbon tab "Ponchevit USM" / panel "Acciones".
- `Ponchevit/Commands/` — thin `IExternalCommand` entrypoints (one per ribbon button). Currently houses placeholder `CommandOne` (retired in Phase 4.8 when `AgregarFamiliaCommand` replaces it).
- `Ponchevit/Composition/Services.cs` — manual composition root. No DI container.
- `Ponchevit/Domain/` — pure C# (no RevitAPI). Sub-folders: `Model`, `Graph` (EmptyBridgeResolver, CodeAssembler), `Query` (PrefixPathQuery, CascadeMenuBuilder), `Catalog` (PartidaCatalog, PartidaHierarchyResolver, PartidaFilter), `Aliases`, `Materials` (IMaterialMappingResolver + substring suggester), `Matching` (IElementMatcher and friends), `Codificacion` (CodificacionSummary record).
- `Ponchevit/Data/` — pure C# (no RevitAPI). Interfaces: `IPartidasRepository`, `ICoveninRulesRepository`, `IMaterialMappingRepository`. `Sqlite/` holds the SQLite-backed implementations of the first two; the third's implementation lives in `Revit/Materials/` because `ExtensibleStorage` is RevitAPI-bound.
- `Ponchevit/Revit/` — the only layer besides `Commands/` that may touch RevitAPI. Includes `SharedParameters/` (CoveninParameters), `SharedParameterWriter.cs`, `ElementTopologyReader.cs`, `Families/` (IFamilyGenerator + MuroGenerator), `Materials/` (ExtensibleStorageMaterialMappingRepository), `Codificacion/` (ProjectInventoryReader), and `Context/IRevitContext.cs`.
- `Ponchevit/Ui/` — WPF; must not call RevitAPI directly. Sub-folders per command surface: `MaterialMapping/`, `AgregarFamilia/` (becomes `PartidaSelectionWindow` shared with Asignar in Phase 5), `AsignarCodigo/`, `Codificacion/`, plus `Common/` (Theme.xaml token sink; `ObservableObject`/`RelayCommand` come from CommunityToolkit.Mvvm — no hand-rolled MVVM helpers).
- `Ponchevit/Infrastructure/Log.cs` — `ILog` + `FileLog` writing to `%AppData%\Ponchevit\log.txt`.
- `Ponchevit/Resources/` — `partidas.db`, `covenin.db`, `SharedParameters.txt` (regenerated from C# constants per ADR 2026-05-31 — GUID source-of-truth), ribbon icons.
- `Ponchevit/manifest/Ponchevit.addin` — Revit add-in manifest with relative `<Assembly>` path. `AddInId` GUID `1bfbb086-06af-4ddf-b84b-99d3fad0366f` must stay stable.
- `Ponchevit/docs/` — authoritative design specs:
  - `architecture.md` — layer discipline, folder boundaries, data stores, storage & persistence, hard constraints.
  - `roadmap.md` — phased plan with checkboxes; single source of truth for done vs. next.
  - `decisions.md` — ADR log capturing deviations and rationale (newest first).
  - `domain-model.md` — Domain layer reference.
  - `data-layer.md` — both SQLite schemas and the in-document ExtensibleStorage mapping schema, side-by-side.
  - `plugin.md` — early UI/workflow spec (predates the design pivot; cross-reference with roadmap + decisions for the current scope).
  - `partidas.md`, `tablas.md` — original source-data descriptions; useful for the Python pipelines.
- `Partidas/`, `Tablas/` (repo root, outside csproj) — Python data-extraction/scraping pipelines that produce the SQLite content. Not built by the solution; treat as separate tooling.
- `.opencode/agent/` — architect / coder / docu_writer subagent definitions for opencode.

## Build / run

- Windows-only build. `Ponchevit.csproj` references `RevitAPI.dll` / `RevitAPIUI.dll` via `HintPath` to `C:\Program Files\Autodesk\Revit 2026\`. Revit 2026 must be installed at that path or the references must be repointed.
- `dotnet build Ponchevit.slnx` builds everything; `dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj` runs the test suite. Filter a single test with `--filter "FullyQualifiedName~YourTestName"`.
- PostBuild step copies the built DLL and `manifest/Ponchevit.addin` to `%AppData%\Autodesk\Revit\Addins\2026\`. Uses `cmd` `copy /Y`, so it only works on Windows (not WSL/Linux dotnet).
- The `<Assembly>` element inside `Ponchevit.addin` is a relative path (`Ponchevit.dll`) so the manifest is portable across machines.
- No CI. Manual verification is loading the add-in in Revit 2026 and exercising the ribbon commands.

## Architecture in one paragraph

Two SQLite data sources ship in `Resources/`: `partidas.db` (~2350-row flat catalog of known partidas — drives the left tree and the right filtered panel) and `covenin.db` (~377k-row Flyweight DAG of column-value rules — drives the central panel's parametric dropdowns and the Asignar auto-prefill). A third store — the per-project material mapping (Revit `ExtensibleStorage`, one `DataStorage` element with a stable `Schema` GUID, serialized `Dictionary<string,string>`) — lives inside each user's .rvt file and travels with it. **The .rvt is the single source of truth for *all* state Ponchevit creates**: codes are written as 4 instance shared parameters on elements, the material mapping is embedded in the document, and Codificación reports are native Revit `ViewSchedule` views (fire-and-forget — the plugin tracks none after creation). There are no sidecar files. The MVP targets local .rvt files; cloud workshare polish (mapping workset placement, sync prompts, borrow-conflict UX) is Post-MVP. See `docs/architecture.md` § "Storage & persistence" for the full storage map and hard constraints (GUID stability, schema evolution).

## Hard layer rule

- `Domain/` and `Data/` MUST NOT reference `RevitAPI` / `RevitAPIUI`. Only `Commands/` and `Revit/` may.
- UI in `Ui/` accesses Revit only through `IRevitContext`.
- Canonical example: `IMaterialMappingRepository` lives in `Data/` (pure C#), but its `ExtensibleStorage`-backed implementation lives in `Revit/Materials/` because Revit's ExtensibleStorage is a RevitAPI feature.

## Adding a ribbon command

1. New class in `Commands/` implementing `IExternalCommand` with `[Transaction(TransactionMode.Manual)]` for write operations or `ReadOnly` for query-only.
2. Register in `App.OnStartup` via `PushButtonData` + `panel.AddItem(...)`. The four MVP commands are: Mapeo de Materiales (Phase 3.5), Agregar Familia (Phase 4.8), Asignar Código (Phase 5.7), Codificación Dashboard (Phase 6.7).
3. Wire dependencies from `Composition/Services.cs`.

## UI conventions

- WPF + `CommunityToolkit.Mvvm` source generators (`[ObservableProperty]`, `[RelayCommand]`). No heavyweight MVVM framework (Prism, ReactiveUI, Caliburn.Micro).
- Raw WPF controls only — no MahApps.Metro, MaterialDesignInXamlToolkit, or commercial control libraries (deferred pending an in-Revit smoke test for assembly conflicts and host-theme behavior).
- `Theme.xaml` is an empty ResourceDictionary — the future token sink for styling.

## Commit conventions

- Conventional Commits with prefixes: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci`.
- Subject line in English, imperative ("Add CodeAssembler 10-digit cap").
- In the commit body when applicable, reference roadmap task IDs (`feat(domain): add CodeAssembler with 10-digit cap (roadmap 1.3)`).
- **Do not commit unless explicitly asked.** The repo owner runs all `git commit` / `git push` / `git add` themselves.
