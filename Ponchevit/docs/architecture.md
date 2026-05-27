# Architecture Reference

## Purpose & non-goals

This document describes the intended *layered* architecture for Ponchevit (Revit 2026 add-in). The goal is a small, stable MVP that is easy to refactor into multiple projects later.

It supports the *Agregar*, *Asignar*, and *Reconocer* workflows, with *Capítulo*/*Subcapítulo*/*Sección* organized into *Partida*; *Muro* is the first *IFamilyGenerator* target.

### Purpose

- Keep `Domain/` and `Data/` independent from Revit so they remain unit-testable and reusable.
- Contain all Revit API usage to `Revit/` (and UI transitions via `IRevitContext`).
- Make ribbon commands thin entrypoints: compose UI/VM, then delegate real work to services.
- Use a manual composition root so the dependency graph is explicit and easy to reason about.

### Non-goals (explicitly NOT doing)

- DI container
- MVVM framework
- EF Core
- Async-everywhere
- Big abstractions
- Installer
- Custom theme

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
│   └── Aliases/                 (IAliasResolver, IdentityAliasResolver)
├── Data/                        (PURE C# — zero RevitAPI references)
│   ├── IPartidasRepository.cs
│   ├── ICoveninRulesRepository.cs
│   └── Sqlite/                  (SqlitePartidasRepository, SqliteCoveninRulesRepository, ConnectionFactory)
├── Revit/                       (the only place besides Commands that touches RevitAPI)
│   ├── SharedParameters/CoveninParameters.cs
│   ├── SharedParameterWriter.cs
│   ├── ElementTopologyReader.cs
│   ├── Families/                (IFamilyGenerator, MuroGenerator)
│   └── Context/IRevitContext.cs
├── Ui/                          (WPF — must not call RevitAPI directly; uses IRevitContext)
│   ├── Common/                  (ObservableObject, RelayCommand, Theme.xaml, FilteredPartidaCollection)
│   ├── AgregarFamilia/          (Phase 3; renamed PartidaSelectionWindow in Phase 4)
│   ├── AsignarCodigo/           (Phase 4 — reuses PartidaSelectionWindow with VM Mode flag)
│   └── ReconocerElemento/       (Phase 5 stretch)
├── Infrastructure/Log.cs        (ILog + FileLog → %AppData%\Ponchevit\log.txt)
├── Resources/                   (partidas.db, covenin.db, SharedParameters.txt, icons)
└── manifest/Ponchevit.addin
```

## Two data sources

Ponchevit ships two SQLite databases side-by-side. They are orthogonal: one is a flat catalog of known codes, the other is a parametric grammar.

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

Parametric grammar (Flyweight DAG) defining, for each prefix, the next column and its admissible values. Drives the **central panel** dynamic dropdowns and **Reconocer Elemento** dimensional matching via `Num_Min`/`Num_Max`/`Unidad`.

| Table | Rows | Loading strategy |
|---|---|---|
| `Covenin_Columnas` | 45 | Eager at startup. |
| `Covenin_Valores` | 379 | Eager at startup. |
| `Covenin_Conexiones` | 376,987 (with 250 empty bridges) | **Lazy, by `Parent_Id`, with in-memory cache.** |

The DAG currently has rules only for Capítulo `E4` (Obras Arquitectónicas), which exactly matches the MVP scope (Muros under E41 Albañilería). Other capítulos will be added later — UI must degrade gracefully when no DAG rules exist for the current prefix.

### How the two are weaved in the UI

| UI region | Driven by |
|---|---|
| Left tree (taxonomy) | `IPartidasRepository` (flat) |
| Central panel (parametric dropdowns) | `ICoveninRulesRepository` (DAG) via `CascadeMenuBuilder` |
| Right panel (known partidas matching prefix) | `PartidaCatalog` (sourced from `IPartidasRepository`), filtered in memory |

## The hard rule

`Domain/` and `Data/` MUST NOT reference `RevitAPI` / `RevitAPIUI`. Justification: unit-testable headlessly, reusable outside Revit, and ports possible later.

## Composition root

`Composition/Services.cs` is instantiated once in `App.OnStartup`. It manually wires:

- `ILog`
- `IAliasResolver`
- `IPartidasRepository`
- `ICoveninRulesRepository`
- `PartidaCatalog`
- `IRevitContext`

Commands resolve dependencies from it. No DI container.

## Patterns we use

- Repository pattern for read-only DB access (`IPartidasRepository` + `ICoveninRulesRepository`).
- Flyweight implicit in the schema (`Covenin_Columnas`/`Covenin_Valores` are shared by `Covenin_Conexiones`).
- Strategy: `IFamilyGenerator` (Muro now; Puerta/Ventana later).
- ExternalEvent for modeless UI → Revit transitions (`IRevitContext.PostExternalEvent`).
- Thin commands: each `IExternalCommand` only constructs a VM/Window and shows it; all work happens through services.
- Two-repository split — flat catalog (`IPartidasRepository`) vs. DAG rules (`ICoveninRulesRepository`) — keeps orthogonal concerns separate and lets each be backed/tested independently.

## Future split

When the MVP stabilizes, fold into a multi-project structure where folder boundaries become mechanical refactors:

- `Ponchevit.Domain.csproj` (netstandard2.0)
- `Ponchevit.Data.Sqlite.csproj` (netstandard2.0)
- `Ponchevit.Revit.csproj` (net8.0-windows)
- `Ponchevit.Ui.csproj` (net8.0-windows)
- `Ponchevit.Tests.csproj`

## How to add a new feature

1. If it’s a new ribbon command, create `Commands/X.cs`.
2. If it queries the DAG, work in `Domain/` only.
3. If it touches the Revit model, isolate it in `Revit/`.
4. Put the UI in `Ui/`. It must call Revit only through `IRevitContext`.
5. Wire dependencies in `Composition/Services.cs`.

Never let Revit API types leak into `Domain/` or `Data/` namespaces.

## Tech choices summary table

| Concern | Choice | Reason |
|---|---|---|
| SQLite access | `Microsoft.Data.Sqlite` | Compatible with manual repository + easy fixture-based tests. |
| UI | WPF + hand-rolled MVVM mini-framework | No MVVM framework dependency; small surface area. |
| Errors | `try/catch` in each command + `TaskDialog` + `ILog` + `Result.Failed` | Keeps failure UX consistent while logging details. |
| Shared parameters | 4 split params (`Capitulo_COVENIN`, `Subcapitulo_COVENIN`, `Seccion_COVENIN`, `Codigo_COVENIN_Completo`) with extras-dictionary support | Native Revit schedule filtering + forward compatibility for future column-value parameters. |
| DB shipping | Two SQLite files in `Resources/` (`partidas.db` flat catalog, `covenin.db` DAG rules) copied beside DLL in `Addins\2026\` via PostBuild; each validated against its own `_meta.schema_version` at startup | Simple installation; each dataset versioned independently. |
