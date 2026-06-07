# Ponchevit Roadmap (MVP Plan)

This phased plan is the single source of truth for what is done vs. what is next.

It tracks the *Agregar*, *Asignar*, *Mapeo de Materiales*, and *Codificación Dashboard* workflows, with codes organized by *Capítulo*/*Subcapítulo*/*Sección* into *Partida*; the first generated element is a *Muro*.

## Phase 0 — Foundation (no feature change; prep the ground)

Exit criteria: `dotnet build Ponchevit.slnx` succeeds; add-in still loads in Revit 2026; "Command One" still works.

- [x] 0.1 — Restructure folders per architecture.md (App.cs, Commands/, Composition/, Infrastructure/, Domain/, Data/, Revit/, Ui/, Resources/). Existing CommandOne stays wired until Phase 4.8. (e912d11)
- [x] 0.2 — Infrastructure/Log.cs: ILog interface + FileLog (writes to %AppData%\Ponchevit\log.txt). (e912d11)
- [x] 0.3 — Composition/Services.cs composition root (wires ILog only for now). (e912d11)
- [x] 0.4 — Rewrite manifest/Ponchevit.addin to use relative path Ponchevit.dll; remove hardcoded username. (e912d11)
- [x] 0.5 — Create Ponchevit.Tests/Ponchevit.Tests.csproj (xUnit, net8.0); add to Ponchevit.slnx. (e912d11)
- [x] 0.6 — Write docs/architecture.md, docs/roadmap.md, docs/decisions.md; update root AGENTS.md. (e912d11)
- [x] 0.7 — Add .editorconfig (C# defaults) and Conventional Commits note in AGENTS.md. (e912d11)


## Phase 1 — Domain + Data (pure C#, zero Revit refs)

Exit criteria: all Domain/Data tests green. No RevitAPI reference anywhere under `Domain/` or `Data/`.

- [x] 1.1 — Domain/Model: DAG types (Columna, Valor, Conexion) + flat catalog types (Capitulo, Subcapitulo, Seccion, Partida) + CodigoCovenin value type (2346b8d)
- [x] 1.2 — Domain/Graph/EmptyBridgeResolver — handles `Codigo_Aportado=""`, inherits Parent_Id. (e371b5d)
- [x] 1.3 — Domain/Graph/CodeAssembler — root→leaf concat, 10-digit firewall, exposes ComputePrefix(connectionId). (e371b5d)
- [x] 1.4 — Domain/Query/PrefixPathQuery — DAG-derived; no dependency on stored mask columns (which don't exist yet). (161e1a0)
- [x] 1.5 — Domain/Query/CascadeMenuBuilder — given partial selection, returns next-level options + remaining required columns (drives central panel). (161e1a0)
- [x] 1.6 — Domain/Catalog/PartidaCatalog — reads the 2081 known partidas from `IPartidasRepository`; attaches Subcapítulo + Sección via PartidaHierarchyResolver; logs and excludes schema anomalies (non-10-digit codes, placeholder codes like `E015xxx5xx`); cached for the Revit session.
- [x] 1.7 — Domain/Catalog/PartidaHierarchyResolver — longest-prefix match of `Partida.codigo` against `Seccion.codigo` (fall back along the prefix chain); pure C#, table-driven.
- [x] 1.8 — Domain/Catalog/PartidaFilter — pure predicate `(selectionState) → IReadOnlyList<Partida>`.
- [x] 1.9 — Domain/Aliases/IAliasResolver + IdentityAliasResolver (MVP passthrough; future SqliteAliasResolver backs the `Covenin_Alias` table when it exists).
- [x] 1.10 — Data/IPartidasRepository interface (read-only) — exposes Capitulos/Subcapitulos/Secciones/Partidas.
- [x] 1.11 — Data/ICoveninRulesRepository interface (read-only) — exposes Columnas, Valores, and lazy `GetConexionesByParent(Id_Conexion?)` for DAG traversal. (99da3c9)
- [x] 1.12 — Data/Sqlite/SqlitePartidasRepository (Microsoft.Data.Sqlite) — eager-loads all four tables from `partidas.db` (small enough; ~2350 total rows). (99da3c9)
- [x] 1.13 — Data/Sqlite/SqliteCoveninRulesRepository — eager-loads `Covenin_Columnas` (45) + `Covenin_Valores` (379) from `covenin.db`; lazy + in-memory cache on `Covenin_Conexiones` keyed by `Parent_Id`. (99da3c9)
- [x] 1.14 — Data/Sqlite/ConnectionFactory — resolves both DB paths beside the DLL (`partidas.db`, `covenin.db`); validates each `_meta.schema_version` row independently and raises a clear error per missing/mismatched DB. (99da3c9)
- [x] 1.15 — xUnit tests for Domain + Data using two in-memory SQLite fixtures (one per schema). Cover: code assembler incl. empty bridges + 10-digit cap, prefix-path correctness, cascade builder, both repositories, PartidaHierarchyResolver longest-prefix logic, schema-anomaly exclusion in catalog load. (99da3c9)
- [x] 1.16 — Write docs/domain-model.md and docs/data-layer.md (the latter documents both `partidas.db` and `covenin.db` schemas + loading strategies side-by-side). (local)

## Phase 2 — Revit adapters + material mapping plumbing

Exit criteria: manual Revit smoke test confirms (a) the 4 shared params bind to all `OST_Model*` categories in a fresh project, and (b) a material mapping written through the plugin survives save/reload of the .rvt file.

- [x] 2.1 — Revit/SharedParameters/CoveninParameters — declare the 4 GUIDs as `static readonly Guid` constants (single source of truth, never changed; see ADR 2026-05-31 — GUID source-of-truth); `EnsureBoundToProject(Document)` binds the 4 params to all `OST_Model*` categories; if `Resources/SharedParameters.txt` is missing, regenerate it from the constants. (local)
- [x] 2.2 — Revit/SharedParameterWriter — writes the 4 params; signature accepts `Dictionary<string,string>` of extras for forward compatibility, but throws `NotImplementedException` if non-empty (see ADR 2026-05-31 — Extras dictionary deferred from MVP); uses caller-supplied Transaction. (local)
- [x] 2.3 — Domain/Materials/IMaterialMappingResolver — pure-C# resolver `(revitMaterialName) → CoveninMaterialValueId?`; passthrough/null when unmapped. Includes a pure substring-suggester helper used by the Mapping UI. (local)
- [x] 2.4 — Data/IMaterialMappingRepository — read/write interface for the per-project mapping (`GetAll`, `Set`, `Remove`, `Clear`). (local)
- [x] 2.5 — Revit/Materials/ExtensibleStorageMaterialMappingRepository — backs `IMaterialMappingRepository` using a Revit `Schema` (stable GUID) + a single `DataStorage` element holding a serialized `Dictionary<string,string>` of revit-material-name → covenin-value-id. (local)
- [x] 2.6 — Revit/ElementTopologyReader — extracts category, layers/materials/thicknesses, dimensions; routes materials through `IMaterialMappingResolver`. (local)
- [x] 2.7 — Revit/Families/IFamilyGenerator + MuroGenerator (CompoundStructure). (local)
- [x] 2.8 — Revit/Context/IRevitContext — wraps UIDocument, exposes selection accessor + ExternalEvent posting. (local)
- [x] 2.9 — xUnit tests for `IMaterialMappingResolver`, the substring suggester, and an in-memory fake of `IMaterialMappingRepository`. The ExtensibleStorage impl is verified by manual smoke test (requires a live `Document`). (local)
- [x] 2.10 — Write docs/revit-integration.md (shared params, ExtensibleStorage schema with the chosen Schema GUID, family generator pattern). (local)

## Phase 3 — Mapeo de Materiales UI

Exit criteria: user opens "Mapeo de Materiales", maps each project material to a Covenin material value (or leaves unmapped), saves, and the mappings persist across .rvt save/reload — including when the file is opened by a different user via cloud workshare.

- [x] 3.1 — Ui/Common/Theme.xaml (empty ResourceDictionary, token sink for future styling). MVVM base types (`ObservableObject`, `RelayCommand`) come from the `CommunityToolkit.Mvvm` NuGet — no hand-rolled helpers (see ADR 2026-05-31 — UI: CommunityToolkit.Mvvm). (local)
- [x] 3.2 — Ui/MaterialMapping/MaterialMappingWindow.xaml + ViewModel — three-column grid (Revit material | Sugerencia | Covenin material dropdown). Lists all project materials. (local)
- [x] 3.3 — Substring-match suggester column powered by the helper in `Domain/Materials` (e.g., name contains "alum" → suggest "Aluminio"); user accepts or overrides. (local)
- [x] 3.4 — Save action → `IRevitContext.PostExternalEvent` → `IMaterialMappingRepository.Set` for changed rows; cancel discards. (local)
- [x] 3.5 — MapeoMaterialesCommand wired in App.OnStartup (ribbon panel "Acciones"). (local)

## Phase 4 — Agregar Familia (first generation slice)

Exit criteria: in Revit 2026, the user creates a code-compliant Muro through the plugin and the 4 shared parameters are populated on the new wall. Material dropdowns surface only Revit materials mapped to the chosen Covenin material.

- [x] 4.1 — Ui/Common/FilteredPartidaCollection. (0dd9cd3)
- [x] 4.2 — Ui/AgregarFamilia/AgregarFamiliaWindow.xaml + ViewModel — three-panel layout. (0dd9cd3)
- [x] 4.3 — Left tree (Capítulo/Subcapítulo/Sección) with Muros enabled, rest disabled via generator registry (`IFamilyGenerator` lookup). (0dd9cd3)
- [x] 4.4 — Central panel dynamic dropdowns driven by `CascadeMenuBuilder`. Material parameters render as a two-stage control: Covenin material → Revit material filtered by `IMaterialMappingResolver`; inline "Mapear material" affordance when nothing maps for the chosen Covenin material. Numeric/range columns show an additional TextBox below the ComboBox only when `NumMin ≠ NumMax`. (0dd9cd3 + post-smoke-test fixes)
- [x] 4.5 — "Subir modelo 3D" button stubbed (not needed for Muros). (0dd9cd3)
- [x] 4.6 — Right panel always populated; `FilteredPartidaCollection` shrinks dynamically as left tree and central panel change. Constructible/unconstructible rows; explicit "Seleccionar Partida" button backfills cascade. Fast in-memory filter (no DB round-trips). (0dd9cd3 + post-smoke-test fixes)
- [x] 4.7 — "Agregar" button → `IRevitContext.PostExternalEvent` → `FamilyGenerationOrchestrator` → `MuroGenerator.Generate` (creates WallType with CompoundStructure from material + thickness) → `SharedParameterWriter` (writes 4 params on WallType). (0dd9cd3 + post-smoke-test fixes)
- [x] 4.8 — AgregarFamiliaCommand wired in App.OnStartup ribbon panel. (post-smoke-test fixes)
- [x] 4.9 — Write docs/ui-patterns.md. (0dd9cd3)

## Phase 5 — Asignar Código (reuses Phase 4 window, adds prefill)

Exit criteria: an existing wall in a project gets tagged with COVENIN params via the plugin; auto-prefill populates the params that can be inferred from category, dimensions, and the material mapping; qualitative params (Mecanismo, Composición, Acabado for walls, etc.) stay empty and visible for the user to fill — no fake confidence.

**Mode behavioral differences (Generate vs Assign):**

| Aspect | Generate mode | Assign mode |
|---|---|---|
| Left tree enabling | Only sections where `IFamilyGenerator.IsGenerable` returns true | All sections enabled |
| Right panel partidas | `CanBeConstructed` flag drives grey-out styling | All partidas selectable — no grey-out |
| Action button | Creates a new family type in the document | Writes shared params onto the target element |
| Target element | None — result is a new WallType / family type | An existing element passed from the Dashboard or pre-selected in Revit before opening |
| Recognition prefill | N/A | Optional — "Reconocer" button runs IElementRecognizer for the target element |

**Element selection rule:** no in-window element picker. The window receives the target element either from the Dashboard (row button passes the FamilyType) or from the ribbon command (user pre-selects in Revit before clicking). If no element is pre-selected when opened from the ribbon, the "Elemento a codificar" field shows a prompt and the Asignar button stays disabled.

- [ ] 5.1 — Rename AgregarFamiliaWindow → PartidaSelectionWindow; VM gains `Mode` (Generate | Assign) and an optional `TargetElement` parameter for Assign mode. Wire the two behavioral differences listed above through the Mode flag.
- [ ] 5.2 — In Assign mode: "Elemento a codificar" field always visible (shows family type name or a greyed "Ningún elemento seleccionado" placeholder). Populated from the passed TargetElement; if null, Asignar button is disabled and a hint explains pre-selection is required.
- [ ] 5.3 — `Revit/Families/IElementRecognizer` interface (mirrors `IFamilyGenerator`; one implementation per element category). `MuroRecognizer` as the reference implementation: reads `ElementTopology` produced by `Revit/ElementTopologyReader`, returns `PrefillResult` — a per-IdColumna map of `{ Value, State }` where State is `AutoFilled | Ambiguous | Undetectable`. Qualitative params always return `Undetectable`.
- [ ] 5.4 — Auto-prefill on element selection in Assign mode. "Reconocer" button runs the matching `IElementRecognizer`; each CascadeRow gains a visual state: auto-filled (highlighted), user-overridden (normal after user changes), undetectable (greyed prompt text). Qualitative params remain empty by design.
- [ ] 5.5 — Prefill report strip beside the central panel (Assign mode only): detected / ambiguous / undetectable counts so the user sees at a glance what still needs choosing.
- [ ] 5.6 — "Asignar" button → `SharedParameterWriter.Write(element, codigo, extras)`. Reads of any incoming code pass through `IAliasResolver` first.
- [ ] 5.7 — AsignarCodigoCommand wired in App.OnStartup (ribbon). Opens with pre-selected element if one is selected in Revit; otherwise opens with TargetElement = null (Assign button disabled). Dashboard row buttons also open this command with the FamilyType pre-filled.

## Phase 6 — Codificación Dashboard (MVP headline)

Exit criteria: user opens the Codificación Dashboard, sees model element **instances** grouped by family type (only types with at least one placed instance — not merely loaded types) with codified/pendiente status and usage quantity, clicks a row to open Asignar prefilled for that family type, and uses "Generar Schedule" to create a native Revit `ViewSchedule` (`COVENIN - Codificación <timestamp>`). Architects export that schedule to budgeting software.

**Dashboard shows per row:**
- Family type name
- COVENIN code (if codified) or "Sin código"
- Usage quantity: Revit native quantity parameter for that category (wall area in m², most other types as instance count); label shows the unit
- Status chip: Codificado / Sin código
- "Asignar código" button → opens Phase 5 PartidaSelectionWindow with the family type pre-filled

**Aggregate header:** "X de Y familias codificadas (ZZ%)" — X codified, Y total used types, % progress.

**Report schedule columns:** 4 COVENIN shared params + family/type name + native Revit quantity field for that category (Area for walls; Count for all others). Quantity fields are native Revit schedule fields — no custom shared param needed.

- [ ] 6.1 — Domain/Codificacion/CodificacionSummary — pure-C# record: per-family-type data (name, codigoCompleto or null, instanceCount, quantityValue, quantityUnit, isCodified).
- [ ] 6.2 — Revit/Codificacion/ProjectInventoryReader — walks the active document using `FilteredElementCollector`; returns only family types that have at least one placed instance; reads the 4 COVENIN shared params; computes quantity via the appropriate Revit built-in parameter per category (walls: `HOST_AREA_COMPUTED`; others: instance count).
- [ ] 6.3 — Ui/Codificacion/CodificacionDashboardWindow.xaml + ViewModel — grouped grid with status filter (Todas | Codificadas | Sin código) and family-name search. Aggregate header shows X/Y codified count and % progress.
- [ ] 6.4 — Each row has an "Asignar código" button → opens PartidaSelectionWindow (Assign mode) with the family type's element pre-filled (uses Phase 5).
- [ ] 6.5 — "Generar Schedule" action: creates a new `ViewSchedule` named `COVENIN - Codificación <timestamp>` scoped to `OST_Model*` categories; columns = 4 COVENIN shared params + family/type name + native Revit quantity field (Area for walls, Count for others). Fire-and-forget (see ADR 2026-05-31 — Fire-and-forget schedules).
- [ ] 6.6 — Manual "Refrescar" button on the dashboard (no live document-event subscription for MVP).
- [ ] 6.7 — CodificacionDashboardCommand wired in App.OnStartup.
- [ ] 6.8 — Write docs/codificacion-dashboard.md.

## Phase 7 — Polish

Exit criteria: icons/theme/docs/decisions are fully cleaned up and Phase 0–6 work is marked as implemented.

- [ ] 7.1 — Round-out tests for any code added in Phases 2–6 that wasn't fully covered.
- [ ] 7.2 — Ribbon icons (16×16 + 32×32 PNGs) for the four commands: Mapeo de Materiales, Agregar Familia, Asignar Código, Codificación Dashboard.
- [ ] 7.3 — Populate Theme.xaml with token defaults so a future theme swap is a one-file change.
- [ ] 7.4 — Mark all docs as "implemented" for their respective phases.
- [ ] 7.5 — Final pass on decisions.md.

## Post-MVP (documented, not scheduled)

- 2–3 reference RFA families per major element type (Muro, Puerta, Ventana) with COVENIN shared parameters pre-applied, plus an "Author your own family" doc so users can extend the catalog without waiting on us.
- Vendor-specific export formats (Excel, PDF) layered on top of the native `ViewSchedule`.
- Wall `Acabado` per-face / per-instance investigation — currently treated as a qualitative per-element param the user picks manually.
- Official/unofficial partida distinction (`Covenin_Partidas_Oficiales` table + `IsOfficial` flag + UI checkbox).
- `Covenin_Alias` table-backed alias resolver to replace the MVP passthrough `IdentityAliasResolver`.
- Cloud workshare polish: dedicated `Ponchevit` workset for the material-mapping `DataStorage` element (or stable workset selection at first creation); post-save "Sync to central?" prompt after Mapeo de Materiales; borrow-conflict UX surfacing in Mapeo and Dashboard.
- Extras-dictionary implementation: when a capítulo with column-value parameters enters scope, dynamically mint stable shared params per extra (deterministic, hash-derived GUIDs) and bind to `OST_Model*` (see ADR 2026-05-31 — Extras dictionary).

## How to use this roadmap

1. Check the boxes as work completes.
2. When a task is done, include the commit SHA next to the checkbox, e.g. `- [x] 1.3 — CodeAssembler (abc1234)`.
3. Any deviation from the plan must be captured with a same-day entry in `docs/decisions.md`.
4. Any agent or human session starts by reading this file and the latest `docs/decisions.md` to understand the current context.
