# Architecture Decision Records (ADRs)

Format:

- Each entry starts with `### YYYY-MM-DD — Title`.
- Then include ~3–10 lines with the following fields:
  - **Decision**
  - **Rationale**
  - **Alternatives considered**
  - **Status** (`Active` | `Superseded` | `Deferred`)

How to add an entry: append at the top (newest first), and commit in the same change that introduces the deviation/decision.

### 2026-05-31 — Phase 3: target framework changed to net8.0-windows; UseWPF enabled

**Decision:** `Ponchevit.csproj` and `Ponchevit.Tests.csproj` both target `net8.0-windows` (previously `net8.0`). `<UseWPF>true</UseWPF>` added to the main project to enable XAML compilation.
**Rationale:** WPF requires the Windows TFM for proper XAML source generation (`InitializeComponent`) and for `System.Windows` APIs to resolve at build time. The project has always been Windows-only (RevitAPI HintPaths point to `C:\Program Files\Autodesk\Revit 2026\`), so this is formalising an existing constraint rather than narrowing scope. The test project needs the matching TFM to reference the main project.
**Alternatives considered:** Keep `net8.0` with manual assembly references to WPF DLLs — fragile and non-standard; rejected. Multi-target `net8.0;net8.0-windows` — unnecessary complexity when the project will never run on non-Windows.
**Status:** Active.

### 2026-05-31 — Phase 2: ExtensibleStorage FindStorage uses GetEntity().IsValid() not GetSchemaGuids()

**Decision:** `ExtensibleStorageMaterialMappingRepository.FindStorage()` locates the `DataStorage` element by calling `ds.GetEntity(schema).IsValid()` on each candidate, rather than a `GetSchemaGuids()` / `Contains()` approach.
**Rationale:** `DataStorage.GetSchemaGuids()` does not exist in the Revit 2026 API. The correct pattern to test whether a `DataStorage` element carries a given schema is to call `GetEntity(schema)` and check `IsValid()`. This is the standard approach in all Revit ExtensibleStorage samples.
**Alternatives considered:** `Element.GetEntitySchemaGuids()` exists on some Revit API versions but was not in the resolved namespace under Revit 2026; using `GetEntity().IsValid()` is equivalent and more broadly documented.
**Status:** Active.

### 2026-05-31 — GUID source-of-truth: hardcoded constants in code; files regenerated from them

**Decision:** the 4 COVENIN shared-parameter GUIDs and the material-mapping `ExtensibleStorage` `Schema` GUID are declared as `static readonly Guid` constants in their respective classes (`Revit/SharedParameters/CoveninParameters` for the shared params; `Revit/Materials/ExtensibleStorageMaterialMappingRepository` for the Schema). `Resources/SharedParameters.txt` is regenerated from the shared-parameter constants on first use if missing. The C# constants are the single source of truth; derived files are reproducible from them.
**Rationale:** the GUID-stability constraint (see "Storage & persistence" in architecture.md) requires these GUIDs to be permanent across all installs and project files forever. Hardcoded constants are easier to inspect during code review, can never drift between code and file representations, are recoverable from accidental file deletion, and produce useful diffs when accidentally touched. The `SharedParameters.txt` file remains shipped as a build output so Revit's parameter-binding tooling can read it natively.
**Alternatives considered (documented for future consideration):**
- **File-as-source-of-truth:** scaffold `Resources/SharedParameters.txt` once by hand or via a one-off Revit-side export tool, commit to source, never regenerate at runtime. "If missing" becomes a fail-loud startup error. *Advantage:* exact byte-for-byte fidelity with whatever Revit's own export tool produces — relevant if the .txt format ever carries subtle metadata our regeneration doesn't reproduce. *Rejected for MVP because:* a deleted file is only recoverable from git, the .txt format is not human-friendly for spot-checking GUIDs, and there's no enforcement that any in-code GUID references stay in sync with the file. Reconsider if Revit ever changes the .txt format in a way our regeneration falls behind on.
**Status:** Active.

### 2026-05-31 — UI: CommunityToolkit.Mvvm for MVVM source generation; no UI control library

**Decision:** add the `CommunityToolkit.Mvvm` NuGet package to the main project and use its `[ObservableProperty]` / `[RelayCommand]` source-generator attributes instead of hand-rolled `ObservableObject`/`RelayCommand` helpers. Keep raw WPF controls — no MahApps.Metro, MaterialDesignInXamlToolkit, DevExpress, Telerik, or Syncfusion. `Theme.xaml` remains an empty ResourceDictionary available for token-based styling later.
**Rationale:** ViewModels in Phases 4–6 will each carry 20+ observable properties (three-panel reactive layouts, per-parameter visual states, grouped/filterable grids). Hand-rolled MVVM would produce ~6 lines of boilerplate per property and obscure VM intent. `CommunityToolkit.Mvvm` is Microsoft-shipped, MIT, mostly source generators (zero runtime weight), trivially removable (expand the generated code by hand and uninstall), and the consensus modern MVVM choice for new WPF/WinUI projects. Critically, it does not touch the visual layer, so it carries no risk of conflicting with Revit 2026's host theming or its loaded WPF assemblies.
**Alternatives considered:** stay pure hand-rolled (rejected — boilerplate scaling concern as VMs grow); add a styled control library such as MahApps.Metro or MaterialDesignInXamlToolkit (deferred — needs an in-Revit smoke test to verify no host-theme conflicts and no assembly-version clashes with Revit's loaded WPF deps); commercial libraries like DevExpress / Telerik / Syncfusion (rejected — cost, very large assemblies, high conflict surface with Revit, wildly oversized for Phase 6's dashboard needs).
**Status:** Active. Amends part of the 2026-05-26 "No DI container, no MVVM framework, no installer" decision — the MVVM-framework half is replaced by this; the DI and installer halves remain Active.

### 2026-05-31 — Extras dictionary: per-extra shared parameter, deferred from MVP

**Decision:** `SharedParameterWriter` accepts a `Dictionary<string,string>` of extra column-value parameters in its signature for forward compatibility, but throws `NotImplementedException` if the dictionary is non-empty for MVP. The eventual implementation will dynamically mint a stable shared parameter (with a deterministic, hash-derived GUID) per column-value name on first use, bound to all `OST_Model*` categories alongside the 4 core params.
**Rationale:** the 4 core shared params cover the MVP scope (Muros under E41). Extras are only needed when other capítulos with column-value parameters enter scope, which is Post-MVP. Per-extra shared params (Path A) preserve native Revit schedule column-filtering on those values; a serialized JSON blob (Path B) would lose that. We commit to Path A but defer the implementation rather than commit to the wrong shape now.
**Alternatives considered:** Path B (serialized blob in a single `Extras_COVENIN` param) — rejected because it kills native schedule column-filtering on those values; implement Path A now without a real use case — rejected as premature.
**Status:** Deferred (writer signature accepts the parameter; implementation is `NotImplementedException` for MVP).

### 2026-05-31 — Codificación schedules are fire-and-forget; plugin tracks none

**Decision:** "Generar Schedule" creates a new `ViewSchedule` each time it is invoked, named `COVENIN - Codificación <timestamp>` to avoid name collisions. The plugin does not look for an existing schedule, does not update one, and maintains no reference to schedules it created. Users are free to rename, restyle, duplicate, or delete schedules at will.
**Rationale:** schedules are user-owned reports. Trying to "update in place" creates surprises (overwrote my styling? deleted my column? changed the filter?) and requires the plugin to track which schedule "belongs" to it. Fire-and-forget makes the action trivially predictable: press the button, get a fresh schedule, do whatever you want with it. Multiple schedules from multiple runs is a feature, not a bug — users can compare snapshots in time or scope.
**Alternatives considered:** update-in-place with style preservation (rejected — complex, surprising, requires tracking); single-schedule with overwrite confirmation (rejected — still requires tracking, still surprising).
**Status:** Active. Simplifies Phase 6.5.

### 2026-05-31 — MVP targets local .rvt; cloud workshare considerations deferred

**Decision:** the MVP is designed and tested against local (non-workshared) Revit files. Worksharing-specific behavior — workset placement for the material-mapping `DataStorage` element, post-save "Sync to central?" prompts, borrow-conflict UX in the Mapeo and Dashboard windows — is captured in Post-MVP but not engineered for in MVP. The existing storage design (instance shared params, `ExtensibleStorage`, `ViewSchedule` elements) consists entirely of standard Revit element types that *will* function in workshared projects, but optimal multi-user UX is not an MVP goal.
**Rationale:** worksharing is an orthogonal axis that doubles testing surface and introduces nondeterminism (borrow timing, sync state, who-owns-what). The MVP needs to prove the coding workflow end-to-end first; multi-user polish layers cleanly on top once that is stable. Phase 2.5 uses the default `DataStorage.Create(Document)` (which is correct for non-worksharing models and harmless for worksharing ones).
**Alternatives considered:** design for workshare from day one (rejected — premature, doubles complexity of every Phase 2 task); explicitly refuse to operate on workshared docs (rejected — overly restrictive, the design works for them just without optimization).
**Status:** Active for MVP. Post-MVP work list captures: dedicated `Ponchevit` workset for the mapping `DataStorage`, post-save sync prompt after Mapeo de Materiales, borrow-conflict UX surfacing in Mapeo and Dashboard.

### 2026-05-31 — Material mapping persisted per-project via ExtensibleStorage

**Decision:** the per-project mapping from Revit material names to Covenin material value IDs lives inside the .rvt as a Revit `ExtensibleStorage` `Schema` + `DataStorage` element. A single entity holds a serialized `Dictionary<string,string>` (revit-material-name → covenin-value-id). Interface `IMaterialMappingRepository` lives in `Data/`; the only RevitAPI-touching implementation, `ExtensibleStorageMaterialMappingRepository`, lives in `Revit/Materials/`.
**Rationale:** .rvt files are typically shared via cloud workshare and updated by multiple people; a sidecar file would drift out of sync or be lost on transfer. ExtensibleStorage is Revit's canonical mechanism for plugin metadata that must travel with the model, requires no shared-parameter pollution of the user-facing parameter UI, and survives save/reload.
**Alternatives considered:** sidecar JSON next to .rvt (rejected — handoff fragility on cloud workshare); custom project parameters (rejected — pollutes parameter UI for plugin-internal data); local SQLite in %AppData% (rejected — not per-project, doesn't travel with the file).
**Status:** Active.

### 2026-05-31 — Reconocer dissolved into Asignar as a prefill step

**Decision:** the standalone "Reconocer Elemento" command from the original roadmap is removed. Element recognition becomes an auto-prefill step inside `Asignar Código`: on element selection, the matcher engine fills the parameters it can infer (category, dimensions, material via the per-project mapping); the user accepts, overrides, or completes the remainder. Qualitative parameters (Mecanismo, Composición, wall Acabado, etc.) are never prefilled — the UI shows them as undetectable rather than guessing.
**Rationale:** Covenin parameters for doors/windows are predominantly qualitative design decisions (3 of 4) that cannot be inferred from Revit geometry. A standalone recognition command with no follow-up action was awkward UX; recognition's real value is as a prefill that reduces typing inside the assign flow. An honest "this is undetectable" surface is more useful than a forced guess users will have to audit anyway.
**Alternatives considered:** keep Reconocer as a separate command (rejected — no follow-up action, no clear value); auto-apply recognized values silently to the element (rejected — wrong inferences would be invisible until a downstream audit caught them).
**Status:** Active. Supersedes the previous Phase 5 of the roadmap.

### 2026-05-31 — Wall `Acabado` treated as qualitative for MVP

**Decision:** the `Acabado` parameter for walls (values: `sencillo`, `obra limpia una cara`, `obra limpia dos caras`) is a qualitative parameter the user selects manually. No per-face / per-instance inference for the MVP.
**Rationale:** the value is per-instance (the same wall type can be `sencillo` in one room and `obra limpia dos caras` in another), so inference would require reading per-instance face-finish metadata and reconciling it with the categorical Covenin vocabulary. That investigation is not blocking the MVP and is captured under Post-MVP in the roadmap.
**Alternatives considered:** infer from CompoundStructure layer finishes (deferred — requires investigation into how Revit exposes per-face finishes vs. type-level structure); ignore the parameter (rejected — it is required for valid E41 codes).
**Status:** Active for MVP. Per-face inference deferred to Post-MVP.

### 2026-05-31 — Codificación Dashboard introduced as the MVP headline feature

**Decision:** a new Phase 6 ships a Codificación Dashboard listing model element instances grouped by family type with codified/pendiente status, click-through to Asignar, and a "Generar/Actualizar Schedule" action that materializes a native Revit `ViewSchedule` of the COVENIN-coded elements.
**Rationale:** without a deliverable artifact, the plugin is a coding tool with no terminal output. The dashboard reframes the plugin's value proposition: it makes the project's coding state visible (gap analysis for the user) and produces the report deliverable (the schedule) that downstream quantity-surveying workflows actually consume.
**Alternatives considered:** ship the coding workflows without any summary view (rejected — leaves the user with no audit/report path); fold the summary into Asignar (rejected — different mode of work, deserves its own surface and ribbon entry).
**Status:** Active.

### 2026-05-31 — Report format is a native Revit ViewSchedule for MVP

**Decision:** the dashboard's "Generar/Actualizar Schedule" action creates or refreshes a Revit `ViewSchedule` (default name `COVENIN - Codificación`) scoped to `OST_Model*` categories, with the 4 COVENIN shared parameters as columns alongside family/type names. No Excel/PDF/CSV export in MVP.
**Rationale:** architects and civil engineers already have their own export workflows from Revit schedules; a plugin-side exporter would duplicate native Revit functionality and add dependency weight (interop libraries, version brittleness). A well-formatted schedule meets the user requirement — make the codified data legible and exportable through tools the team already uses.
**Alternatives considered:** Excel interop (rejected — heavy dependency, version brittleness, duplicates native export); PDF generation (rejected — same); CSV writer (rejected — Revit schedules already export to delimited text). All three remain Post-MVP options.
**Status:** Active.

### 2026-05-31 — Reference family catalog scoped to 2–3 per type, deferred to Post-MVP

**Decision:** the MVP ships no preconfigured RFA families. Post-MVP, ship 2–3 reference RFA families per major element type (Muro, Puerta, Ventana) with COVENIN shared parameters pre-applied, plus an "Author your own family" doc so users can extend the catalog themselves.
**Rationale:** RFA files are Revit-version-locked (2026 today, 2027 next year, ongoing maintenance forever); curating "good enough" defaults is itself a product effort; distribution and discoverability add complexity. Ship the tooling that lets users author their own families first; expand a seed catalog only after the core workflows are validated by real use.
**Alternatives considered:** comprehensive bundled catalog in MVP (rejected — tarpit); no catalog at all (rejected — leaves new users without any starting reference even after MVP stabilizes).
**Status:** Deferred (Post-MVP).

### 2026-05-26 — Two-database strategy (partidas.db + covenin.db)

**Decision:** ship two SQLite databases side-by-side in `Resources/`: `partidas.db` (flat catalog of known partidas; 10 capítulos, 46 subcapítulos, 190 secciones, 2081 partidas) and `covenin.db` (DAG rules; 45 columns, 379 values, ~377k connections). Two read-only repository interfaces, `IPartidasRepository` and `ICoveninRulesRepository`, back the domain layer.
**Rationale:** the two datasets have orthogonal shapes and orthogonal purposes (static taxonomy vs. parametric grammar); a single repository would conflate concerns and make either one harder to evolve. Source pipelines in `Partidas/` and `Tablas/` also produce them independently.
**Alternatives considered:** single merged DB (rejected — couples unrelated schemas and lifecycles); flat-only (rejected — loses the parametric central-panel UX and dimensional matching); DAG-only (rejected — no canonical list of known codes).
**Status:** Active. Supersedes the earlier single-DB shipping ADR (which is updated below to plural).

### 2026-05-26 — Lazy loading of `Covenin_Conexiones`

**Decision:** at startup, eager-load `Covenin_Columnas` (45) and `Covenin_Valores` (379) into memory. Do NOT eager-load `Covenin_Conexiones` (~377k rows). Query it on demand by `Parent_Id` and cache resolved subtrees in an in-memory dictionary for the Revit session.
**Rationale:** 377k rows × ~5 columns is heavy for a startup load and is not needed all at once — DAG traversal is naturally local-by-Parent_Id; caching prevents repeated DB hits as the user drills the central panel.
**Alternatives considered:** eager-load everything into a graph (rejected — startup cost, memory pressure); no cache, query every time (rejected — UI latency during cascade).
**Status:** Active.

### 2026-05-26 — Partida → Sección link computed via longest-prefix match

**Decision:** `Partidas.codigo_partida` does not store an FK to `Secciones`. At catalog load, `PartidaHierarchyResolver` resolves each partida's Sección by longest-prefix match of `codigo_partida` against `Secciones.codigo`, falling back along the prefix chain if no exact match. The same logic resolves the Subcapítulo from the matched Sección.
**Rationale:** the source dataset is shaped this way; computing the link at load time is O(n·log m) over small tables (~2081 × 190) and avoids requiring schema changes upstream.
**Alternatives considered:** require an FK in the source DB (rejected — would force the Python pipeline to commit to one mapping); compute on every query (rejected — wasteful, the result is stable for the session).
**Status:** Active.

### 2026-05-26 — Schema anomaly handling (log and exclude)

**Decision:** at catalog load, partidas whose `codigo_partida` is not 10 digits, and sections whose `codigo` contains literal placeholder substrings (e.g. `xxx`), are logged as warnings to `%AppData%\Ponchevit\log.txt` and excluded from the in-memory catalog. The plugin never fails to start because of source-data anomalies.
**Rationale:** known issues in the current source: 3 codes are 9 digits, 4 are 11 digits, and `Secciones` contains a placeholder `E015xxx5xx`. We want visibility for cleanup in the Python pipeline without blocking work.
**Alternatives considered:** include them silently (rejected — masks real upstream bugs); reject load (rejected — blocks all use over fixable upstream cleanup); attempt auto-fix (rejected — out of scope, fragile).
**Status:** Active.

### 2026-05-26 — Reuse PartidaSelectionWindow for Agregar and Asignar

**Decision:** single WPF window with a VM `Mode` (Generate | Assign) toggle.
**Rationale:** user requirement "must be identical to Agregar Familias"; avoids duplicating three-panel layout.
**Alternatives considered:** two separate windows (rejected — duplication).
**Status:** Active.

### 2026-05-26 — Right panel always populated, shrinks dynamically

**Decision:** `PartidaCatalog` eager-loads all valid partidas (~2081 known) at first window open; `PartidaFilter` is a pure in-memory predicate.
**Rationale:** user requirement for fast permissive filtering as left tree + central panel change.
**Alternatives considered:** lazy DB queries per filter change (rejected — latency); pagination (rejected — overkill at this scale).
**Status:** Active.

### 2026-05-26 — Official/unofficial partida distinction deferred post-MVP

**Decision:** MVP treats all enumerable partidas uniformly; no IsOfficial flag in `Partida`, no UI checkbox, no DB schema for it.
**Rationale:** user opted to keep MVP scope tight; revisit if real users encounter combinatorial garbage.
**Reactivation path:** add `Covenin_Partidas_Oficiales(codigo)` table + `IOfficialPartidaSource` interface + `Partida.IsOfficial` + UI checkbox.
**Status:** Deferred.

### 2026-05-26 — Alias resolution interface day one, table deferred

**Decision:** `IAliasResolver` exists from Phase 1.8; MVP uses `IdentityAliasResolver` (passthrough). Future: SqliteAliasResolver backed by `Covenin_Alias(codigo_erroneo, codigo_canonico)` table once the user populates it.
**Rationale:** known typos in source document; cheaper to wire the interface now than retrofit callers later.
**Alternatives considered:** hardcoded map (rejected — not scalable); no abstraction (rejected — retrofit cost).
**Status:** Active (passthrough); table Deferred.

### 2026-05-26 — Masks not stored — computed from DAG

**Decision:** PrefixPathQuery derives Capítulo/Subcapítulo/Sección prefixes by traversing Covenin_Conexiones and concatenating Codigo_Aportado, no dependency on a stored mask column.
**Rationale:** user confirms mask columns don't exist yet; will be added later as optimization.
**Alternatives considered:** wait for masks (rejected — blocks work); cache derived masks aggressively (deferred).
**Status:** Active.

### 2026-05-26 — Shared parameters split into 4 fields

**Decision:** `Capitulo_COVENIN`, `Subcapitulo_COVENIN`, `Seccion_COVENIN`, `Codigo_COVENIN_Completo` — all Text, Instance, bound to all `OST_Model*` categories.
**Rationale:** enables native Revit schedule filtering.
**Alternatives considered:** single combined field (rejected — schedule UX); per-column extras (deferred to future via SharedParameterWriter extras dict).
**Status:** Active.

### 2026-05-26 — Databases shipped beside DLL, each validated against its own schema_version.

**Decision:** `Resources/partidas.db` and `Resources/covenin.db` each copied to `Addins\2026\` alongside `Ponchevit.dll` via PostBuild; each carries its own _meta.schema_version row; mismatch on either refuses load with TaskDialog.
**Rationale:** simplest install, single-source-of-truth versioning.
**Alternatives considered:** embedded resource extracted to AppData (rejected — update friction); user-pointed path (rejected — fragility).
**Status:** Active.

### 2026-05-26 — No DI container, no MVVM framework, no installer

**Decision:** hand-rolled composition root + ~80 LOC MVVM helpers + PostBuild copy.
**Rationale:** codebase is tiny; adding these later is trivial, removing them later is not.
**Status:** Active for DI (still hand-rolled) and installer (still PostBuild copy). The MVVM-helpers half is superseded by the 2026-05-31 UI ADR — `CommunityToolkit.Mvvm` source generators replace the hand-rolled `ObservableObject`/`RelayCommand`.

### 2026-05-26 — Conventional Commits

**Decision:** use `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci` prefixes.
**Rationale:** standardizes history; tool-friendly.
**Status:** Active.

### 2026-05-26 — Docs in English, domain terms in Spanish

**Decision:** prose in English; preserve Spanish for Capítulo, Subcapítulo, Sección, Partida, Muro, Agregar, Asignar, Reconocer, and any norm-specific term.
**Rationale:** consistent with existing docs/plugin.md, partidas.md, tablas.md.
**Status:** Active.
