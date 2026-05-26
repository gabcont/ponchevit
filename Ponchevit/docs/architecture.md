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
│   ├── Catalog/                 (PartidaCatalog, PartidaFilter)
│   └── Aliases/                 (IAliasResolver, IdentityAliasResolver)
├── Data/                        (PURE C# — zero RevitAPI references)
│   ├── ICoveninRepository.cs
│   └── Sqlite/                  (SqliteCoveninRepository, ConnectionFactory)
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
├── Resources/                   (covenin.db, SharedParameters.txt, icons)
└── manifest/Ponchevit.addin
```

## The hard rule

`Domain/` and `Data/` MUST NOT reference `RevitAPI` / `RevitAPIUI`. Justification: unit-testable headlessly, reusable outside Revit, and ports possible later.

## Composition root

`Composition/Services.cs` is instantiated once in `App.OnStartup`. It manually wires:

- `ILog`
- `IAliasResolver`
- `ICoveninRepository`
- `PartidaCatalog`
- `IRevitContext`

Commands resolve dependencies from it. No DI container.

## Patterns we use

- Repository pattern for read-only DB access (`ICoveninRepository`).
- Flyweight implicit in the schema (`Covenin_Columnas`/`Covenin_Valores` are shared by `Covenin_Conexiones`).
- Strategy: `IFamilyGenerator` (Muro now; Puerta/Ventana later).
- ExternalEvent for modeless UI → Revit transitions (`IRevitContext.PostExternalEvent`).
- Thin commands: each `IExternalCommand` only constructs a VM/Window and shows it; all work happens through services.

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
| DB shipping | `Resources/covenin.db` copied beside DLL in `Addins\2026\` + `_meta.schema_version` validation at startup | Simple installation and explicit schema version gating. |
