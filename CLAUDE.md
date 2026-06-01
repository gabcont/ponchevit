# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start every session

Read `Ponchevit/docs/roadmap.md` first. It is the single source of truth for what is done and what is next. Update it with the commit SHA when a task completes. Log any plan deviation in `Ponchevit/docs/decisions.md` on the same day.

## Build and test

```powershell
# Build the solution (Windows only; requires Revit 2026 at the default install path)
dotnet build Ponchevit.slnx

# Run all tests
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj

# Run a single test by name
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj --filter "FullyQualifiedName~YourTestName"
```

PostBuild automatically copies `Ponchevit.dll` and `manifest/Ponchevit.addin` to `%AppData%\Autodesk\Revit\Addins\2026\`. Manual smoke-test in Revit 2026 is the only integration verification path — there is no CI.

## Project layout

```
Ponchevit.slnx                   ← solution root
Ponchevit/Ponchevit.csproj       ← only buildable project (net8.0)
Ponchevit.Tests/                 ← xUnit tests (net8.0, references main project)
Ponchevit/docs/                  ← authoritative design specs (read before changing behavior)
Partidas/, Tablas/               ← Python data-extraction pipelines (not part of the solution)
```

Inside `Ponchevit/`:

```
App.cs                           ← IExternalApplication entrypoint, ribbon wiring
Commands/                        ← thin IExternalCommand entrypoints
Composition/Services.cs          ← manual composition root (no DI container)
Domain/                          ← pure C#, zero RevitAPI references
  Model/                         ← Columna, Valor, Conexion, Partida, CodigoCovenin, etc.
  Graph/                         ← EmptyBridgeResolver, CodeAssembler
  Query/                         ← PrefixPathQuery, CascadeMenuBuilder
  Catalog/                       ← PartidaCatalog, PartidaHierarchyResolver, PartidaFilter
  Aliases/                       ← IAliasResolver, IdentityAliasResolver
  Materials/                     ← IMaterialMappingResolver (pure C#)
  Matching/                      ← IElementMatcher, CategoryMatcher, DimensionalRangeMatcher
  Codificacion/                  ← CodificacionSummary record
Data/                            ← pure C#, zero RevitAPI references
  IPartidasRepository.cs
  ICoveninRulesRepository.cs
  IMaterialMappingRepository.cs
  Sqlite/                        ← SqlitePartidasRepository, SqliteCoveninRulesRepository, ConnectionFactory
Revit/                           ← only layer besides Commands that may use RevitAPI
  SharedParameters/
  Families/                      ← IFamilyGenerator, MuroGenerator
  Materials/                     ← ExtensibleStorageMaterialMappingRepository
  Codificacion/                  ← ProjectInventoryReader
  Context/IRevitContext.cs
Ui/                              ← WPF; must not call RevitAPI directly; uses IRevitContext
  MaterialMapping/, AgregarFamilia/, AsignarCodigo/, Codificacion/
Infrastructure/Log.cs            ← ILog + FileLog → %AppData%\Ponchevit\log.txt
Resources/                       ← partidas.db, covenin.db, SharedParameters.txt, icons
manifest/Ponchevit.addin         ← AddInId GUID 1bfbb086-06af-4ddf-b84b-99d3fad0366f (must stay stable)
```

## Architecture — three data stores

**`partidas.db`** — flat catalog (~2350 rows across `Capitulos`, `Subcapitulos`, `Secciones`, `Partidas`). Eagerly loaded. Drives the left navigation tree and the right panel (always-populated, filtered in memory by `PartidaFilter`).

**`covenin.db`** — DAG rules (`Covenin_Columnas` 45 rows, `Covenin_Valores` 379 rows, `Covenin_Conexiones` ~377k rows). Columnas + Valores eager-loaded at startup; Conexiones **lazy-loaded by `Parent_Id`** with in-memory cache per session. Drives the central-panel dynamic dropdowns via `CascadeMenuBuilder`. Currently rules only exist for Capítulo E4; UI must degrade gracefully for other capítulos.

**Material mapping (per-project, Revit `ExtensibleStorage`)** — user-curated map from Revit material names to Covenin material value IDs. Lives inside the .rvt as a single `DataStorage` element with a stable `Schema` GUID, serialized as `Dictionary<string,string>`. Read/write through `IMaterialMappingRepository`; the only RevitAPI-touching implementation sits in `Revit/Materials/`. Travels with the .rvt over cloud workshare — no sidecar.

`Partida → Sección` links are computed at catalog load via longest-prefix match (`PartidaHierarchyResolver`) — no FK exists in the schema. Schema anomalies (non-10-digit codes, placeholder substrings) are logged and excluded; they never block startup.

The codification report is a native Revit `ViewSchedule` named `COVENIN - Codificación <timestamp>`, fire-and-forget per click — the plugin tracks nothing after creation, users rename/restyle/delete at will. Vendor-specific exports (Excel/PDF) are Post-MVP.

**Source of truth:** the .rvt is the only place state lives. Element codes (4 shared params), material mapping (`ExtensibleStorage`), and Codificación schedules (Revit Views) all sit inside the document. Dashboard state and the family inventory shown in the Dashboard are in-memory derivations, not persisted. Only the addin's logs (`%AppData%\Ponchevit\log.txt`) live outside the .rvt. **MVP is local-only**; cloud workshare polish (workset placement, sync prompts, borrow-conflict UX) is Post-MVP.

## Hard layer rule

`Domain/` and `Data/` **must not** reference `RevitAPI` or `RevitAPIUI`. `Ui/` accesses Revit only through `IRevitContext`. Commands and `Revit/` are the only allowed RevitAPI consumers. The `ExtensibleStorage`-backed mapping repository is the canonical example: interface in `Data/`, implementation in `Revit/Materials/`.

## Adding a ribbon command

1. New class in `Commands/` implementing `IExternalCommand` with `[Transaction(TransactionMode.Manual)]` for write operations or `ReadOnly` for query-only.
2. Register in `App.OnStartup` via `PushButtonData` + `panel.AddItem(...)`.
3. Wire dependencies from `Composition/Services.cs`.

## Conventions

- **Language:** docs in English; domain terms stay in Spanish (`Capítulo`, `Subcapítulo`, `Sección`, `Partida`, `Agregar`, `Asignar`, `Muro`, `Codificación`, `Mapeo de Materiales`, etc.).
- **Commits:** Conventional Commits — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci`. Subject in English, imperative. Reference roadmap task IDs in the body (e.g., `feat(domain): add CodeAssembler (roadmap 1.3)`).
- **No new frameworks or DI containers** without an explicit request — the composition root is hand-rolled by design.
- **Docs over code:** when `docs/` and code diverge, the docs describe the intended target — flag the gap rather than silently re-deriving it.
