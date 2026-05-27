# Ponchevit Roadmap (MVP Plan)

This phased plan is the single source of truth for what is done vs. what is next.

It tracks the *Agregar*, *Asignar*, and *Reconocer* workflows, with codes organized by *Capítulo*/*Subcapítulo*/*Sección* into *Partida*; the first generated element is a *Muro*.

## Phase 0 — Foundation (no feature change; prep the ground)

Exit criteria: `dotnet build Ponchevit.slnx` succeeds; add-in still loads in Revit 2026; "Command One" still works.

- [x] 0.1 — Restructure folders per architecture.md (App.cs, Commands/, Composition/, Infrastructure/, Domain/, Data/, Revit/, Ui/, Resources/). Existing CommandOne stays wired until Phase 3.7. (e912d11)
- [x] 0.2 — Infrastructure/Log.cs: ILog interface + FileLog (writes to %AppData%\Ponchevit\log.txt). (e912d11)
- [x] 0.3 — Composition/Services.cs composition root (wires ILog only for now). (e912d11)
- [x] 0.4 — Rewrite manifest/Ponchevit.addin to use relative path Ponchevit.dll; remove hardcoded username. (e912d11)
- [x] 0.5 — Create Ponchevit.Tests/Ponchevit.Tests.csproj (xUnit, net8.0); add to Ponchevit.slnx. (e912d11)
- [x] 0.6 — Write docs/architecture.md, docs/roadmap.md, docs/decisions.md; update root AGENTS.md. (e912d11)
- [x] 0.7 — Add .editorconfig (C# defaults) and Conventional Commits note in AGENTS.md. (e912d11)


## Phase 1 — Domain + Data (pure C#, zero Revit refs)

Exit criteria: all Domain/Data tests green. No RevitAPI reference anywhere under `Domain/` or `Data/`.

- [ ] 1.1 — Domain/Model: DAG types (Columna, Valor, Conexion) + flat catalog types (Capitulo, Subcapitulo, Seccion, Partida) + CodigoCovenin value type (parses e.g. E411011015 → Capítulo/Subcapítulo/Sección).
- [ ] 1.2 — Domain/Graph/EmptyBridgeResolver — handles `Codigo_Aportado=""`, inherits Parent_Id.
- [ ] 1.3 — Domain/Graph/CodeAssembler — root→leaf concat, 10-digit firewall, exposes ComputePrefix(connectionId).
- [ ] 1.4 — Domain/Query/PrefixPathQuery — DAG-derived; no dependency on stored mask columns (which don't exist yet).
- [ ] 1.5 — Domain/Query/CascadeMenuBuilder — given partial selection, returns next-level options + remaining required columns (drives central panel).
- [ ] 1.6 — Domain/Catalog/PartidaCatalog — reads the 2081 known partidas from `IPartidasRepository`; attaches Subcapítulo + Sección via PartidaHierarchyResolver; logs and excludes schema anomalies (non-10-digit codes, placeholder codes like `E015xxx5xx`); cached for the Revit session.
- [ ] 1.7 — Domain/Catalog/PartidaHierarchyResolver — longest-prefix match of `Partida.codigo` against `Seccion.codigo` (fall back along the prefix chain); pure C#, table-driven.
- [ ] 1.8 — Domain/Catalog/PartidaFilter — pure predicate `(selectionState) → IReadOnlyList<Partida>`.
- [ ] 1.9 — Domain/Aliases/IAliasResolver + IdentityAliasResolver (MVP passthrough; future SqliteAliasResolver backs the `Covenin_Alias` table when it exists).
- [ ] 1.10 — Data/IPartidasRepository interface (read-only) — exposes Capitulos/Subcapitulos/Secciones/Partidas.
- [ ] 1.11 — Data/ICoveninRulesRepository interface (read-only) — exposes Columnas, Valores, and lazy `GetConexionesByParent(Id_Conexion?)` for DAG traversal.
- [ ] 1.12 — Data/Sqlite/SqlitePartidasRepository (Microsoft.Data.Sqlite) — eager-loads all four tables from `partidas.db` (small enough; ~2350 total rows).
- [ ] 1.13 — Data/Sqlite/SqliteCoveninRulesRepository — eager-loads `Covenin_Columnas` (45) + `Covenin_Valores` (379) from `covenin.db`; lazy + in-memory cache on `Covenin_Conexiones` keyed by `Parent_Id`.
- [ ] 1.14 — Data/Sqlite/ConnectionFactory — resolves both DB paths beside the DLL (`partidas.db`, `covenin.db`); validates each `_meta.schema_version` row independently and raises a clear error per missing/mismatched DB.
- [ ] 1.15 — xUnit tests for Domain + Data using two in-memory SQLite fixtures (one per schema). Cover: code assembler incl. empty bridges + 10-digit cap, prefix-path correctness, cascade builder, both repositories, PartidaHierarchyResolver longest-prefix logic, schema-anomaly exclusion in catalog load.
- [ ] 1.16 — Write docs/domain-model.md and docs/data-layer.md (the latter documents both `partidas.db` and `covenin.db` schemas + loading strategies side-by-side).

## Phase 2 — Revit adapters

Exit criteria: manual Revit smoke test confirms shared params appear in a fresh project bound to all model categories.

- [ ] 2.1 — Revit/SharedParameters/CoveninParameters — GUID constants, definition file generation, `EnsureBoundToProject(Document)` binding to all `OST_Model*` categories; generates `Resources/SharedParameters.txt` if missing.
- [ ] 2.2 — Revit/SharedParameterWriter — writes the 4 params; accepts `Dictionary<string,string>` of extras for future column-value parameters; uses caller-supplied Transaction.
- [ ] 2.3 — Revit/ElementTopologyReader — extracts category, layers/materials/thicknesses, dimensions.
- [ ] 2.4 — Revit/Families/IFamilyGenerator + MuroGenerator (CompoundStructure).
- [ ] 2.5 — Revit/Context/IRevitContext — wraps UIDocument, exposes selection accessor + ExternalEvent posting.
- [ ] 2.6 — Write docs/revit-integration.md.

## Phase 3 — Agregar Familia (first vertical slice)

Exit criteria: in Revit 2026, user creates a code-compliant Muro through the plugin and sees the 4 shared parameters populated on the new wall.

- [ ] 3.1 — Ui/Common: ObservableObject, RelayCommand, Theme.xaml (empty ResourceDictionary), FilteredPartidaCollection.
- [ ] 3.2 — AgregarFamiliaWindow.xaml + ViewModel — three-panel layout.
- [ ] 3.3 — Left tree (Capítulo/Subcapítulo/Sección) with Muros enabled, rest disabled via generator registry (`IFamilyGenerator` lookup).
- [ ] 3.4 — Central panel dynamic dropdowns driven by CascadeMenuBuilder; "Subir modelo 3D" button stubbed (not needed for Muros).
- [ ] 3.5 — Right panel always populated; FilteredPartidaCollection shrinks dynamically as left tree and central panel change. Fast in-memory filter (no DB round-trips).
- [ ] 3.6 — "Agregar" button → IRevitContext.PostExternalEvent → MuroGenerator → SharedParameterWriter.
- [ ] 3.7 — Replace CommandOne ribbon button with AgregarFamiliaCommand; wire in App.OnStartup.
- [ ] 3.8 — Write docs/ui-patterns.md.

## Phase 4 — Asignar Código (reuses Phase 3 window)

Exit criteria: existing wall in a project gets tagged with COVENIN params via the plugin.

- [ ] 4.1 — Rename AgregarFamiliaWindow → PartidaSelectionWindow; VM gains Mode (Generate | Assign).
- [ ] 4.2 — In Assign mode: all sections enabled, "Elemento a codificar" field visible. If a Revit element is preselected when command runs, populate it; else open in-window element picker sub-dialog.
- [ ] 4.3 — "Asignar" button → SharedParameterWriter.Write(element, codigo, extras). Reads of any incoming code pass through IAliasResolver first.
- [ ] 4.4 — AsignarCodigoCommand wired to ribbon.

## Phase 5 — Reconocer Elemento (stretch, only if Phases 0–4 complete with time)

Exit criteria: topology-aware scanning (if implemented) routes matched Partida(s) to the Assign flow.

- [ ] 5.1 — Domain/Matching/IElementMatcher + CategoryMatcher + DimensionalRangeMatcher (uses Num_Min/Num_Max/Unidad from Valores) (sourced via `ICoveninRulesRepository`).
- [ ] 5.2 — Ui/ReconocerElemento/ comparative scanner window.
- [ ] 5.3 — ReconocerElementoCommand entrypoint; hands matched partida off to the Assign flow.
- [ ] 5.4 — Topology-match strategy stays designed-for but unimplemented unless time permits; record a decisions.md entry if added.

## Phase 6 — Polish

Exit criteria: icons/theme/docs/decisions are fully cleaned up and Phase 0–5 work is marked as implemented.

- [ ] 6.1 — Round-out tests for any code added in Phases 2–5 that wasn't fully covered.
- [ ] 6.2 — Ribbon icons (16×16 + 32×32 PNGs).
- [ ] 6.3 — Populate Theme.xaml with token defaults so a future theme swap is a one-file change.
- [ ] 6.4 — Mark all docs as "implemented" for their respective phases.
- [ ] 6.5 — Final pass on decisions.md.

## How to use this roadmap

1. Check the boxes as work completes.
2. When a task is done, include the commit SHA next to the checkbox, e.g. `- [x] 1.3 — CodeAssembler (abc1234)`.
3. Any deviation from the plan must be captured with a same-day entry in `docs/decisions.md`.
4. Any agent or human session starts by reading this file and the latest `docs/decisions.md` to understand the current context.
