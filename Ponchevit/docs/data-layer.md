# Data Layer Strategy (Phase 1.16)

This document explains the intended **Data layer** design. The Data layer is pure C# (no Revit types) and provides the repositories that feed the Domain.

Ponchevit intentionally uses **two separate SQLite databases**, one for the flat taxonomy catalog and one for the parametric DAG rules.

---

## 1) Two SQLite databases

### 1.1 `Resources/partidas.db` — flat catalog (taxonomy)

Purpose:

- authoritative list of known `Capitulo` / `Subcapitulo` / `Seccion` / terminal `Partida` codes,
- drives the **left tree** and provides the base list filtered for the right panel.

Tables (conceptual):

- `Capitulos` (10 rows)
- `Subcapitulos` (46 rows)
- `Secciones` (190 rows)
- `Partidas` (2081 rows)

Note:

- `Partidas` stores the terminal code plus `unidad` and `descripcion` and a `capitulo` title string.
- The link from a `codigo_partida` to the matching `Sección/Subcapítulo` is computed in the Domain using `PartidaHierarchyResolver` (no stored FK required).

### 1.2 `Resources/covenin.db` — DAG rules (parametric grammar)

Purpose:

- parametric rule engine backing the central dropdown cascade,
- used to drive prefix-path derivation and (later) dimensional matching for `Reconocer Elemento`.

Tables (conceptual):

- `Covenin_Columnas` (45 rows)
- `Covenin_Valores` (379 rows)
- `Covenin_Conexiones` (~376,987 rows)

Key modeling detail:

- `Covenin_Conexiones` is the adjacency list, with traversal direction defined by `Parent_Id`.
- Each edge contributes a code fragment via `Codigo_Aportado` (empty fragments represent empty bridges).

---

## 2) ConnectionFactory and schema validation (`_meta`)

`ConnectionFactory` centralizes SQLite connection creation for both DBs:

- It selects the DB base path beside the running assembly (`Ponchevit.dll`).
- It exposes:
  - `CreatePartidasConnection()` → `partidas.db`
  - `CreateCoveninConnection()` → `covenin.db`

Before returning a connection, it validates schema compatibility via a `_meta` table:

1. Confirms `_meta` exists in `sqlite_master`.
2. Reads the first `_meta.schema_version` row.
3. Compares it to `ExpectedSchemaVersion` (currently `1`).

If `_meta` is missing or the version mismatches, the factory throws an `InvalidOperationException` with a clear DB-specific message.

This “fail fast” approach prevents the plugin from running against stale/incorrect datasets.

---

## 3) Raw ADO.NET approach using `Microsoft.Data.Sqlite`

The Data layer uses the low-level `Microsoft.Data.Sqlite` APIs (connections, commands, and data readers). There is no ORM and no query abstraction layer.

The repositories follow the intended loading strategy:

- **`SqlitePartidasRepository`**: eager-load all taxonomy tables.
- **`SqliteCoveninRulesRepository`**: eager-load everything except `Covenin_Conexiones` (load connections lazily, with in-memory caching).

### 3.1 `SqlitePartidasRepository` (eager-load)

`SqlitePartidasRepository` constructs in-memory lists for:

- `Capitulos`
- `Subcapitulos`
- `Secciones`
- `Partidas`

Each list is populated in the constructor by running explicit `SELECT` queries using:

- `SqliteConnection` created by `ConnectionFactory`
- `SqliteCommand`
- `SqliteDataReader`

The implementation loads each table via a dedicated method and stores results in read-only fields used by `GetCapitulos()`, `GetSubcapitulos()`, `GetSecciones()`, and `GetPartidas()`.

### 3.2 `SqliteCoveninRulesRepository` (lazy connections + cache)

`SqliteCoveninRulesRepository` uses a split strategy:

**Eager-loaded at startup**

- `Covenin_Columnas` → `_columnas` dictionary
- `Covenin_Valores` → `_valores` dictionary

**Lazy-loaded on demand**

- `Covenin_Conexiones` is *not* loaded entirely.
- `GetConexionesByParent(parentId)`:
  - for `parentId` null/empty: loads root connections once and stores them in `_rootConexiones`.
  - for a specific `parentId`: checks `_conexionesCache`, otherwise queries DB and caches the result.

DB filtering logic for `Covenin_Conexiones`:

- root edges: `Parent_Id IS NULL OR Parent_Id = ''`
- non-root edges: `Parent_Id = @parentId`

This prevents a heavy startup load while keeping UI navigation fast via per-`Parent_Id` caching.
