# Architecture Decision Records (ADRs)

Format:

- Each entry starts with `### YYYY-MM-DD — Title`.
- Then include ~3–10 lines with the following fields:
  - **Decision**
  - **Rationale**
  - **Alternatives considered**
  - **Status** (`Active` | `Superseded` | `Deferred`)

How to add an entry: append at the top (newest first), and commit in the same change that introduces the deviation/decision.

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
**Status:** Active.

### 2026-05-26 — Conventional Commits

**Decision:** use `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci` prefixes.
**Rationale:** standardizes history; tool-friendly.
**Status:** Active.

### 2026-05-26 — Docs in English, domain terms in Spanish

**Decision:** prose in English; preserve Spanish for Capítulo, Subcapítulo, Sección, Partida, Muro, Agregar, Asignar, Reconocer, and any norm-specific term.
**Rationale:** consistent with existing docs/plugin.md, partidas.md, tablas.md.
**Status:** Active.
