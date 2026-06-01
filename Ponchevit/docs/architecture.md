# Architecture Reference

## Purpose & non-goals

This document describes the intended *layered* architecture for Ponchevit (Revit 2026 add-in). The goal is a small, stable MVP that is easy to refactor into multiple projects later.

It supports the *Agregar*, *Asignar*, *Mapeo de Materiales*, and *Codificación Dashboard* workflows, with *Capítulo*/*Subcapítulo*/*Sección* organized into *Partida*; *Muro* is the first *IFamilyGenerator* target.

### Purpose

- Keep `Domain/` and `Data/` independent from Revit so they remain unit-testable and reusable.
- Contain all Revit API usage to `Revit/` (and UI transitions via `IRevitContext`).
- Make ribbon commands thin entrypoints: compose UI/VM, then delegate real work to services.
- Use a manual composition root so the dependency graph is explicit and easy to reason about.

### Non-goals (explicitly NOT doing)

- DI container
- Heavyweight MVVM frameworks (Prism, ReactiveUI, Caliburn.Micro) — `CommunityToolkit.Mvvm` source generators only, no runtime framework
- WPF control libraries (MahApps.Metro, MaterialDesignInXamlToolkit, DevExpress, Telerik, Syncfusion) — raw WPF controls only; deferred pending an in-Revit smoke test for assembly conflicts
- EF Core
- Async-everywhere
- Big abstractions
- Installer
- Custom theme
- Vendor-specific report exports (Excel, PDF) — the native Revit `ViewSchedule` is the MVP report; vendor formats are Post-MVP

## Folder/namespace layout

The codebase is organized by layer boundary. Mirror the following layout:

```text
Ponchevit/
├── App.cs                       (IExternalApplication entrypoint, ribbon wiring)
├── Commands/                    (IExternalCommand entrypoints — thin)
├── Composition/Services.cs      (manual composition root)
├── Domain/                      (PURE C# — zero RevitAPI references)
│   ├── Model/                   (Columna, Valor, Conexion, Partida, CodigoCovenin)
│   ├── Graph/                   (EmptyBridgeResolver, CodeAssembler)
│   ├── Query/                   (PrefixPathQuery, CascadeMenuBuilder)
│   ├── Catalog/                 (PartidaCatalog, PartidaFilter, PartidaHierarchyResolver)
│   ├── Aliases/                 (IAliasResolver, IdentityAliasResolver)
│   ├── Materials/               (IMaterialMappingResolver, substring suggester)
│   ├── Matching/                (IElementMatcher, CategoryMatcher, DimensionalRangeMatcher)
│   └── Codificacion/            (CodificacionSummary record)
├── Data/                        (PURE C# — zero RevitAPI references)
│   ├── IPartidasRepository.cs
│   ├── ICoveninRulesRepository.cs
│   ├── IMaterialMappingRepository.cs
│   └── Sqlite/                  (SqlitePartidasRepository, SqliteCoveninRulesRepository, ConnectionFactory)
├── Revit/                       (the only place besides Commands that touches RevitAPI)
│   ├── SharedParameters/CoveninParameters.cs
│   ├── SharedParameterWriter.cs
│   ├── ElementTopologyReader.cs
│   ├── Materials/               (ExtensibleStorageMaterialMappingRepository)
│   ├── Codificacion/            (ProjectInventoryReader)
│   ├── Families/                (IFamilyGenerator, MuroGenerator)
│   └── Context/IRevitContext.cs
├── Ui/                          (WPF — must not call RevitAPI directly; uses IRevitContext)
│   ├── Common/                  (ObservableObject, RelayCommand, Theme.xaml, FilteredPartidaCollection)
│   ├── MaterialMapping/         (Phase 3 — MaterialMappingWindow)
│   ├── AgregarFamilia/          (Phase 4; renamed PartidaSelectionWindow in Phase 5)
│   ├── AsignarCodigo/           (Phase 5 — reuses PartidaSelectionWindow with VM Mode flag + prefill)
│   └── Codificacion/            (Phase 6 — CodificacionDashboardWindow)
├── Infrastructure/Log.cs        (ILog + FileLog → %AppData%\Ponchevit\log.txt)
├── Resources/                   (partidas.db, covenin.db, SharedParameters.txt, icons)
└── manifest/Ponchevit.addin
```

## Data stores

Ponchevit reads from three orthogonal data stores. The first two ship as SQLite files in `Resources/`; the third lives inside each user's .rvt file as Revit `ExtensibleStorage`.

### `Resources/partidas.db` (~468 KB) — Partidas catalog (flat)

Authoritative list of *known* COVENIN partidas and their hierarchy. Drives the **left tree** (Capítulo/Subcapítulo/Sección) and the **right panel** (always-populated list, filtered as the user narrows).

| Table | Rows | Purpose |
|---|---|---|
| `Capitulos` | 10 | Top-level groupings (E0–E9). |
| `Subcapitulos` | 46 | Nested under Capítulo. |
| `Secciones` | 190 | Nested under Subcapítulo; variable code length. |
| `Partidas` | 2081 | The terminal codes; PK is `codigo_partida` (variable length, mostly 10 digits). |

The `Partidas` table only stores a denormalized `capitulo` title string. The link to the matching `Sección` is *computed* at catalog load via longest-prefix match — see `PartidaHierarchyResolver`.

### `Resources/covenin.db` (~25.6 MB) — Covenin rules (DAG)

Parametric grammar (Flyweight DAG) defining, for each prefix, the next column and its admissible values. Drives the **central panel** dynamic dropdowns and the **Asignar Código auto-prefill** dimensional matching via `Num_Min`/`Num_Max`/`Unidad`.

| Table | Rows | Loading strategy |
|---|---|---|
| `Covenin_Columnas` | 45 | Eager at startup. |
| `Covenin_Valores` | 379 | Eager at startup. |
| `Covenin_Conexiones` | 376,987 (with 250 empty bridges) | **Lazy, by `Parent_Id`, with in-memory cache.** |

The DAG currently has rules only for Capítulo `E4` (Obras Arquitectónicas), which exactly matches the MVP scope (Muros under E41 Albañilería). Other capítulos will be added later — UI must degrade gracefully when no DAG rules exist for the current prefix.

### Per-project material mapping — Revit `ExtensibleStorage` (in-document)

User-curated mapping from Revit material names to Covenin material value IDs. Persisted as a single `DataStorage` element inside the active document with a stable `Schema` GUID, holding a serialized `Dictionary<string,string>`. Travels with the .rvt over cloud workshare — no sidecar file to lose during handoff.

Read/write is exposed through `Data/IMaterialMappingRepository`. The only RevitAPI-touching implementation, `Revit/Materials/ExtensibleStorageMaterialMappingRepository`, sits under `Revit/` because `ExtensibleStorage` is a RevitAPI feature; keeping the interface in `Data/` preserves the rule that Domain code never depends on Revit. The mapping is consumed by `Domain/Materials/IMaterialMappingResolver`, which `ElementTopologyReader` and the Asignar-Código prefill both go through when interpreting a Revit material.

### How the data stores are weaved in the UI

| UI region | Driven by |
|---|---|
| Left tree (taxonomy) | `IPartidasRepository` (flat) |
| Central panel (parametric dropdowns) | `ICoveninRulesRepository` (DAG) via `CascadeMenuBuilder` |
| Right panel (known partidas matching prefix) | `PartidaCatalog` (sourced from `IPartidasRepository`), filtered in memory |
| Material dropdowns (Agregar / Asignar) | Covenin material values from `ICoveninRulesRepository`; Revit material list filtered by `IMaterialMappingResolver` |
| Material Mapping window | `IMaterialMappingRepository` (read/write) + project material list from `IRevitContext` |
| Codificación Dashboard | `Revit/Codificacion/ProjectInventoryReader` (model walk) projected into `Domain/Codificacion/CodificacionSummary` |
| Codificación Schedule view | Newly created `ViewSchedule` (named `COVENIN - Codificación <timestamp>`) on each "Generar Schedule" press; columns = the 4 COVENIN shared parameters; plugin tracks none after creation |

## Storage & persistence

The .rvt file is the single source of truth for every piece of state Ponchevit creates. There are no sidecar files, no external databases, and no per-user state on disk beyond the addin's own logs. This is what makes the design transferable: open the .rvt on any machine with the addin installed and the picture is identical.

### Storage map

| Data | Where it lives | Persistence | Updated by |
|---|---|---|---|
| COVENIN codes on elements | 4 instance shared parameters per element (`Capitulo_COVENIN`, `Subcapitulo_COVENIN`, `Seccion_COVENIN`, `Codigo_COVENIN_Completo`), bound to all `OST_Model*` categories | Inside the .rvt as native element params | `SharedParameterWriter` during Agregar / Asignar |
| Shared-parameter definitions | Project-wide bindings (saved into the .rvt at first bind); definitions sourced from `Resources/SharedParameters.txt` shipped with the addin DLL | Inside the .rvt after first `EnsureBoundToProject` invocation | `CoveninParameters.EnsureBoundToProject(Document)` (idempotent) |
| Material mapping | One `Schema` (stable GUID) + one `DataStorage` element with a serialized `Dictionary<string,string>` | Inside the .rvt via Revit `ExtensibleStorage` | Mapeo de Materiales window via `IMaterialMappingRepository` |
| Codificación schedules | Real `ViewSchedule` elements (`COVENIN - Codificación <timestamp>`), columns = the 4 shared params; always live against current element values | Inside the .rvt as Revit View elements; fire-and-forget — see ADR 2026-05-31 | Dashboard "Generar Schedule" action |
| Dashboard state (filter, search, sort) | In-memory only, per window-open | Not persisted; opens fresh with defaults each time | N/A |
| Family inventory shown in Dashboard | In-memory only, derived from a document walk via `ProjectInventoryReader` | Not persisted; rebuilt on Refresh | N/A |
| Addin logs | `%AppData%\Ponchevit\log.txt` | Local per-user, per-machine | `FileLog` |

### Hard constraints

- **The 4 shared-parameter GUIDs and the material-mapping `Schema` GUID MUST NEVER change.** They live as `static readonly Guid` constants in `Revit/SharedParameters/CoveninParameters` and `Revit/Materials/ExtensibleStorageMaterialMappingRepository` (see ADR 2026-05-31 — GUID source-of-truth); `Resources/SharedParameters.txt` is regenerated from the constants. Changing any of these GUIDs orphans every existing .rvt that uses Ponchevit — old values become invisible to the plugin while remaining as orphan project params under their original names.
- **Mapping payload evolution stays inside the schema.** When the mapping payload needs to grow (e.g., add a `lastEditedBy` field per entry), the right path is to switch the serialized payload to a versioned JSON object (`{ "version": 2, "entries": [...] }`) and write a forward-compatible reader — *not* mint a new ExtensibleStorage `Schema` with a new GUID.

### MVP scope: local .rvt only

The MVP is designed and tested against local (non-workshared) Revit files. All storage mechanisms used are standard Revit element types and *will* work in workshared models, but multi-user UX considerations are not engineered for in MVP. See the Post-MVP section of `roadmap.md` for the specific cloud-workshare follow-ups (mapping `DataStorage` workset placement, post-save sync prompt, borrow-conflict UX in Mapeo and Dashboard).

## The hard rule

`Domain/` and `Data/` MUST NOT reference `RevitAPI` / `RevitAPIUI`. Justification: unit-testable headlessly, reusable outside Revit, and ports possible later. The `ExtensibleStorage`-backed implementation of `IMaterialMappingRepository` is the canonical example — the interface lives in `Data/`, the Revit-specific implementation lives in `Revit/Materials/`.

## Composition root

`Composition/Services.cs` is instantiated once in `App.OnStartup`. It manually wires:

- `ILog`
- `IAliasResolver`
- `IPartidasRepository`
- `ICoveninRulesRepository`
- `IMaterialMappingRepository` (Revit-backed via ExtensibleStorage)
- `IMaterialMappingResolver`
- `PartidaCatalog`
- `IRevitContext`
- `IFamilyGenerator` registry (Muro for MVP)
- `IElementMatcher` set (category, dimensional range)

Commands resolve dependencies from it. No DI container.

## Patterns we use

- Repository pattern for read-only DB access (`IPartidasRepository` + `ICoveninRulesRepository`) and read/write per-project metadata (`IMaterialMappingRepository`).
- Flyweight implicit in the schema (`Covenin_Columnas`/`Covenin_Valores` are shared by `Covenin_Conexiones`).
- Strategy: `IFamilyGenerator` (Muro now; Puerta/Ventana later); `IElementMatcher` (category, dimensional range now; more later).
- ExternalEvent for modeless UI → Revit transitions (`IRevitContext.PostExternalEvent`).
- Thin commands: each `IExternalCommand` only constructs a VM/Window and shows it; all work happens through services.
- Three-store data plane: flat catalog (SQLite), DAG rules (SQLite), per-project material mapping (Revit `ExtensibleStorage`). Each is behind its own interface in `Data/` so Domain code stays pure C#.

## Future split

When the MVP stabilizes, fold into a multi-project structure where folder boundaries become mechanical refactors:

- `Ponchevit.Domain.csproj` (netstandard2.0)
- `Ponchevit.Data.Sqlite.csproj` (netstandard2.0)
- `Ponchevit.Revit.csproj` (net8.0-windows)
- `Ponchevit.Ui.csproj` (net8.0-windows)
- `Ponchevit.Tests.csproj`

## How to add a new feature

1. If it's a new ribbon command, create `Commands/X.cs`.
2. If it queries the DAG or the catalog, work in `Domain/` only.
3. If it touches the Revit model or document-stored metadata, isolate it in `Revit/`.
4. Put the UI in `Ui/`. It must call Revit only through `IRevitContext`.
5. Wire dependencies in `Composition/Services.cs`.

Never let Revit API types leak into `Domain/` or `Data/` namespaces.

## Tech choices summary table

| Concern | Choice | Reason |
|---|---|---|
| SQLite access | `Microsoft.Data.Sqlite` | Compatible with manual repository + easy fixture-based tests. |
| UI | WPF + `CommunityToolkit.Mvvm` source generators (no runtime framework, no control library) | Source generators eliminate `INotifyPropertyChanged` boilerplate without adding runtime weight or framework lock-in. Microsoft-maintained, MIT, trivially removable (expand generated code by hand and uninstall). Raw WPF controls avoid conflicts with Revit's loaded WPF assemblies. |
| Errors | `try/catch` in each command + `TaskDialog` + `ILog` + `Result.Failed` | Keeps failure UX consistent while logging details. |
| Shared parameters | 4 split params (`Capitulo_COVENIN`, `Subcapitulo_COVENIN`, `Seccion_COVENIN`, `Codigo_COVENIN_Completo`) with extras-dictionary support | Native Revit schedule filtering + forward compatibility for future column-value parameters. |
| Per-project material mapping | Revit `ExtensibleStorage` (Schema + single DataStorage); serialized `Dictionary<string,string>` | Travels with the .rvt over cloud workshare; no sidecar handoff fragility; no shared-param pollution for plugin-internal metadata. |
| Codification report | Native Revit `ViewSchedule` (`COVENIN - Codificación <timestamp>`) scoped to `OST_Model*`; fire-and-forget per click | Plugin does not track or update existing schedules — users own them; multiple snapshots in one project is a feature. Excel/PDF/CSV remain Post-MVP. |
| DB shipping | Two SQLite files in `Resources/` (`partidas.db` flat catalog, `covenin.db` DAG rules) copied beside DLL in `Addins\2026\` via PostBuild; each validated against its own `_meta.schema_version` at startup | Simple installation; each dataset versioned independently. |
