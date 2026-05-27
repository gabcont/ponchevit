# Domain Model (Phase 1.16)

This document describes the intended **Domain layer** design for Ponchevit. The Domain layer is pure C# (no Revit types, no EF/ORM), and it models:

1) a **flat catalog** for the taxonomy (**Capítulo / Subcapítulo / Sección / Partida**) and
2) a **DAG** for the parametric rule graph (**Columna / Valor / Conexión**).

On top of those models, the Domain provides:

- graph algorithms to assemble and reason about **COVENIN codes**,
- query builders that drive the cascade UI, and
- catalog services that resolve hierarchy links and filter anomalies.

---

## 1) Flat catalog models

The flat catalog is represented by four immutable record models:

- `Capitulo`: root-level group (e.g. *E4*), with `(Id, Codigo, Titulo)`.
- `Subcapitulo`: nested under a `Capitulo`, with `(Id, CapituloId, Codigo, Titulo)`.
- `Seccion`: nested under a `Subcapitulo`, with `(Id, CapituloId, SubcapituloId, Codigo, Titulo)`.
- `Partida`: terminal item with `(CodigoPartida, Unidad, Descripcion, Capitulo, Subcapitulo?, Seccion?)`.

**Important:** `Partida` does *not* rely on stored foreign keys in the DB. The Domain resolves the `Capitulo/Subcapitulo/Seccion` attachment at catalog load time (see `PartidaHierarchyResolver`).

---

## 2) DAG models (Flyweight graph)

The parametric rules are represented as a Directed Acyclic Graph (DAG) using three immutable record models:

- `Columna (IdColumna, Nombre)`: logical variable/type in the grammar (e.g., *MATERIAL*, *ESPESOR*).
- `Valor (IdValor, DescripcionUi, IdColumna, NumMin?, NumMax?, Unidad?)`: semantic option; may carry numeric range constraints.
- `Conexion (IdConexion, ParentId?, CodigoAportado, IdColumna, IdValorAsociado?)`: an edge in the DAG.

This shape intentionally matches the SQLite schema:

- columns/values are reused across many connections (flyweight semantics),
- connections are the adjacency list and define the navigation direction through `ParentId`, and
- each connection contributes (or not) a fragment of the final code via `CodigoAportado`.

---

## 3) `CodigoCovenin` — strongly typed code value

`CodigoCovenin` is a small value type wrapper around a string:

- `Value`: the raw code fragment.
- `Length`, `IsEmpty`: convenience properties.
- `ToString()` and implicit conversions to/from `string`.
- `StartsWith(prefix)`: case-insensitive prefix helper.

This wrapper makes it harder to mix partial strings with real code values, and it keeps all code assembly/prefix logic in one place.

---

## 4) Graph algorithms

### 4.1 `EmptyBridgeResolver`

Some DAG edges represent **empty bridges** (puentes vacíos) that do not contribute digits to the final code.

`EmptyBridgeResolver` formalizes this as:

- `IsEmptyBridge(Conexion)`: true when `Conexion.CodigoAportado` is null/empty.
- `FilterBridges(path)`: removes empty-bridge edges from a path when you only want digit-contributing connections.

### 4.2 `CodeAssembler`

`CodeAssembler` turns a *DAG path* (a sequence of `Conexion`) into a `CodigoCovenin`.

Core behavior:

- `Assemble(path)` concatenates each connection’s `CodigoAportado` from left to right.
- Empty contributions are skipped.
- A hard **10-digit firewall** enforces `MaxCodeLength = 10`:
  - once the assembled string reaches 10 characters, any further contributions are ignored (or truncated for the last fragment).

Additional helper:

- `ComputePrefix(path, targetConnectionId)` assembles only up to the connection whose `IdConexion` matches `targetConnectionId`.

---

## 5) Query builders (Domain services)

### 5.1 `PrefixPathQuery`

`PrefixPathQuery` derives hierarchical prefixes (**Capítulo / Subcapítulo / Sección**) from a DAG path.

The current implementation assumes the first three logical steps in the path correspond to these levels:

- Capítulo = assemble first 1 connection
- Subcapítulo = assemble first 2 connections
- Sección = assemble first 3 connections

It returns a `PrefixPathResult(CodigoCovenin Capitulo, CodigoCovenin Subcapitulo, CodigoCovenin Seccion)`.

### 5.2 `CascadeMenuBuilder`

`CascadeMenuBuilder` builds the next UI level for the cascading dropdowns.

Given a `parentConnectionId`:

1. It queries `ICoveninRulesRepository.GetConexionesByParent(parentConnectionId)`.
2. If no children exist, it returns `null`.
3. It expects all children of that step to belong to the same `Columna` (uses the first child’s `IdColumna`).
4. For each `Conexion`, it creates a `MenuOption`:
   - `IdConexion = Conexion.IdConexion`
   - `IdValor = Conexion.IdValorAsociado` (nullable)
   - `CodigoAportado = Conexion.CodigoAportado`
   - `Label` =
     - `Valor.DescripcionUi` when `IdValorAsociado` is present, otherwise
     - `CodigoAportado` as a fallback.

The result is a `MenuLevel(Columna, Options)` ready for the central panel.

---

## 6) Catalog resolution services

### 6.1 `PartidaCatalog`

`PartidaCatalog` is the Domain entry point that builds the in-memory list of valid `Partida` values.

It does:

1. Loads `Capitulos`, `Subcapitulos`, and `Secciones` from `IPartidasRepository`.
2. Creates `PartidaHierarchyResolver`.
3. Iterates all repository `Partida` rows and applies:
   - **Schema anomaly filtering** (`IsAnomaly`):
     - blank/whitespace `CodigoPartida`
     - code length != 10
     - placeholder marker: `CodigoPartida` contains the character `x` (case-insensitive)
   - **Hierarchy resolution** via `PartidaHierarchyResolver.Resolve(p.CodigoPartida)`.

Logging and exclusion behavior:

- If `IsAnomaly` returns true, it logs a warning and excludes that partida.
- If the hierarchy cannot be resolved (`Capitulo` is null), it logs another warning and excludes the partida.

The logger is `ILog` (backed by FileLog in Infrastructure), so warnings are emitted to the runtime log file.

### 6.2 `PartidaHierarchyResolver`

`PartidaHierarchyResolver` attaches `Partida` codes to `Sección/Subcapítulo/Capítulo` using **longest-prefix match**.

Implementation approach:

- It sorts `Capitulo`, `Subcapitulo`, and `Seccion` lists by **descending code length**.
- For a given `codigoPartida`, it finds the first match where `codigoPartida.StartsWith(c.Codigo)`.

Because longer codes are checked first, the first match is the “longest prefix” match for each level.
