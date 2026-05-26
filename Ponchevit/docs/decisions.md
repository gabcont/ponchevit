# Architecture Decision Records (ADRs)

Format:

- Each entry starts with `### YYYY-MM-DD — Title`.
- Then include ~3–10 lines with the following fields:
  - **Decision**
  - **Rationale**
  - **Alternatives considered**
  - **Status** (`Active` | `Superseded` | `Deferred`)

How to add an entry: append at the top (newest first), and commit in the same change that introduces the deviation/decision.

### 2026-05-26 — Reuse PartidaSelectionWindow for Agregar and Asignar

**Decision:** single WPF window with a VM `Mode` (Generate | Assign) toggle.
**Rationale:** user requirement "must be identical to Agregar Familias"; avoids duplicating three-panel layout.
**Alternatives considered:** two separate windows (rejected — duplication).
**Status:** Active.

### 2026-05-26 — Right panel always populated, shrinks dynamically

**Decision:** `PartidaCatalog` eager-loads all valid partidas (≤~1000) at first window open; `PartidaFilter` is a pure in-memory predicate.
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

### 2026-05-26 — DB shipped beside DLL, validated against schema_version

**Decision:** `Resources/covenin.db` copied to `Addins\2026\` alongside `Ponchevit.dll` via PostBuild; `_meta.schema_version` row required, mismatch refuses load with TaskDialog.
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
