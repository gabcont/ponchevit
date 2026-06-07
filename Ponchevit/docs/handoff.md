# Ponchevit — Session Handoff

Last updated: 2026-06-06. Read this alongside `roadmap.md` (current state) and `decisions.md` (all ADRs).

---

## What this project is

Ponchevit is a Revit 2026 add-in that helps architects and civil engineers apply **COVENIN** construction-norm codes to elements in a Revit model. It was built for the Venezuelan COVENIN E4 standard (structural elements). The plugin lives entirely inside the `.rvt` file — no sidecar files, no external databases at runtime.

**Three commands (MVP):**
1. **Mapeo de Materiales** — maps Revit project materials to COVENIN material value IDs. Stored in ExtensibleStorage inside the `.rvt`.
2. **Agregar Familia** — lets the user pick a COVENIN code via a three-panel UI, then creates a code-compliant WallType with compound structure and 4 shared parameters.
3. **Asignar Código** (Phase 5, not yet built) — tags an existing element with a COVENIN code; prefills from element geometry/materials.
4. **Codificación Dashboard** (Phase 6, not yet built) — summary view + native Revit schedule generator.

---

## Current status — what is done

### Phases 0–3: Complete and stable

| Phase | Summary |
|---|---|
| 0 | Folder structure, log infrastructure, composition root, test project, docs |
| 1 | Full domain model (Columna, Valor, Conexion, Partida, CodigoCovenin), DAG graph (EmptyBridgeResolver, CodeAssembler), query layer (PrefixPathQuery, CascadeMenuBuilder), catalog (PartidaCatalog, PartidaHierarchyResolver, PartidaFilter, PartidaConstructibilityResolver), both SQLite repositories, 92 unit tests |
| 2 | Revit adapters: CoveninParameters (4 shared params), SharedParameterWriter, ExtensibleStorageMaterialMappingRepository, IFamilyGenerator + MuroGenerator, IRevitContext |
| 3 | Mapeo de Materiales window (three-column grid, substring suggester, save via PostExternalEvent) |

### Phase 4: Complete — smoke-tested and fixed

The **Agregar Familia** workflow is fully implemented and smoke-tested. All sub-items are done:

- **Left tree** (Capítulo → Subcapítulo → Sección): enabled nodes are those whose code prefix matches a registered `IFamilyGenerator`. Currently only E41x nodes are enabled (MuroGenerator supports `"E41"`).
- **Central panel** with path-based seeding (from tree node selection), cascading COVENIN dropdowns, two-stage material control (COVENIN value → Revit material), range TextBox for `NumMin ≠ NumMax` options, live code display, and status feedback.
- **Right panel** with constructibility-annotated partidas, greyed unconstructible rows, and explicit "Seleccionar Partida" backfill button.
- **Agregar** → `PostExternalEvent` → `FamilyGenerationOrchestrator` → MuroGenerator → WallType with CompoundStructure + 4 shared params on the type.

### Phase 5 and Phase 6: Not started

The next work is Phase 5 (Asignar Código) and Phase 6 (Codificación Dashboard). See `roadmap.md` for their exit criteria and task lists.

---

## Architecture overview

### Hard layer rules

```
Domain/   ──→  zero RevitAPI, zero Data (can call Data interfaces)
Data/     ──→  zero RevitAPI
Ui/       ──→  zero RevitAPI direct calls; uses IRevitContext only
Revit/    ──→  may use RevitAPI freely
Commands/ ──→  may use RevitAPI freely; thin entrypoints only
```

Violations of this rule break the test suite (tests reference `Domain/` and `Data/` without Revit).

### Three data stores

| Store | Location | Access |
|---|---|---|
| `partidas.db` | `Resources/` (SQLite, ~2350 rows) | Eager-loaded by `SqlitePartidasRepository` at startup |
| `covenin.db` | `Resources/` (SQLite, ~377k connections) | Columnas + Valores eager; Conexiones lazy-loaded by Parent_Id with in-memory cache |
| Material mapping | Inside `.rvt` (ExtensibleStorage Schema GUID `2E4B6E3A-…`) | Read/write via `IMaterialMappingRepository`; implementation in `Revit/Materials/` |

### Key singletons (Services.cs)

```
Services
├── Log                         FileLog → %AppData%\Ponchevit\log.txt
├── RevitContext                RevitContextImpl (UIApplication wrapper + ExternalEvent)
├── PartidasRepository          SqlitePartidasRepository
├── CoveninRulesRepository      SqliteCoveninRulesRepository
├── PartidaCatalog              Eager-loaded; GetPartidas() → IReadOnlyList<Partida>
├── ConstructibilityResolver    One-time DFS at startup; O(1) IsConstructible + GetPath
├── AliasResolver               IdentityAliasResolver (passthrough for MVP)
├── MaterialMappingRepository   ExtensibleStorageMaterialMappingRepository
├── MaterialMappingResolver     MaterialMappingResolver (pure-C# lookup)
├── FamilyGenerators            IFamilyGenerator[] { MuroGenerator }
├── GenerationOrchestrator      FamilyGenerationOrchestrator (owns Revit transactions)
├── ProjectMaterialQuery        ProjectMaterialQuery (FilteredElementCollector wrapper)
└── HierarchyResolver           PartidaHierarchyResolver (built once from catalog tables)
```

`Services.Build()` is called once in `App.OnStartup`. Commands call `services.RevitContext.Attach(commandData.Application)` before doing anything Revit-specific.

---

## Agregar Familia — full data flow

```
User selects tree node (e.g. "Muros E411")
    → VM.SelectTreeNodeCommand
    → _filteredPartidas.ApplyFilter(capCodigo, subCodigo, secCodigo)
    → SeedCascadeFromTreeNode(node)
        → ConstructibilityResolver.GetPath(seedPartida.CodigoPartida)  [pre-computed path]
        → SeedRowsFromPath(nodeCode, fullPath)                          [add seeded rows; stop at nodeCode]
        → AppendNextCascadeLevel(lastConnId)                            [first user-selectable row]
    → UpdateCodeDisplay()                                               ["E411XXXXXX"]
    → RebuildDisplayItems()                                             [right panel]

User changes a cascade dropdown
    → OnCascadeSelectionChanged(changedRow)
    → Rebuild _selectedConnectionPath from rows above + new row
    → Truncate rows below changedRow
    → If material column: RefreshRevitMaterials(row)                    [Stage 2 dropdown]
    → AppendNextCascadeLevel(changedRow.SelectedOption.IdConexion)
    → UpdateCodeDisplay() + UpdateRightPanelFromPath() + UpdateCanAgregar()

User clicks "Seleccionar Partida" (with a constructible partida highlighted)
    → ConfirmPartidaCommand
    → BackfillCascadeFromPath(GetPath(selectedPartida.CodigoPartida))   [SetSelectedSilently on each row]
    → UpdateCodeDisplay() → "E4110705FF" (full 10-digit code)
    → CanAgregar = true

User clicks "Agregar"
    → AgregarCommand (CanExecute = CanAgregar)
    → BuildGeneratorInput()
        SelectedValores: IdColumna → Valor (full object, has NumMin/NumMax/Unidad)
        NumericValues:   IdColumna → double in metres (only for range rows with typed value)
        Descripcion:     from PartidaCatalog lookup on assembled code
        Codigo:          CodigoCovenin(assembledCode)
        Capitulo/Subcapitulo/Seccion: titles from HierarchyResolver.Resolve(assembledCode)
    → _revitContext.PostExternalEvent(doc => orchestrator.Generate(doc, generator, input))
        [Revit main thread:]
        CoveninParameters.EnsureBoundToProject(doc)                     [Transaction 1: bind 4 params]
        Transaction("Agregar Familia COVENIN")                          [Transaction 2:]
            MuroGenerator.Generate(doc, input)
                Step 1: resolve Revit material
                    allMappings = _materialMappingRepo.GetAll()          [Dict<revitName, coveninValueId>]
                    find entry where .Value == selectedValor.IdValor     [inverted lookup]
                    FindRevitMaterialId(doc, mappingEntry.Key)           [FilteredElementCollector]
                Step 2: resolve thickness
                    if NumericValues has the column → use it (metres)
                    else find first SelectedValor with NumMin != null and not a material → use NumMin
                    convert to feet via UnitUtils.ConvertToInternalUnits
                Step 3: ResolveWallType(doc, "COVENIN {code} — {desc}")
                    check existing WallType with that name; if none, Duplicate the first Basic type
                Step 4: newWallType.SetCompoundStructure(
                    CompoundStructure.CreateSimpleCompoundStructure([
                        new CompoundStructureLayer(thicknessInFeet,
                            MaterialFunctionAssignment.Structure,
                            revitMaterialElementId)
                    ]))
                Step 5: SharedParameterWriter.Write(newWallType, codigo, capitulo, subcapitulo, seccion)
            t.Commit()
    → Dispatcher.Invoke: StatusMessage = "Familia creada correctamente." (or error)
```

---

## Key files — quick reference

| File | Role |
|---|---|
| `App.cs` | `IExternalApplication`; wires ribbon buttons in `OnStartup`; calls `Services.Build()` |
| `Composition/Services.cs` | Manual composition root; built once; consumed via `App.Services` |
| `Commands/AgregarFamiliaCommand.cs` | Thin entrypoint; attaches RevitContext; creates VM + Window |
| `Commands/MapeoMaterialesCommand.cs` | Same pattern for material mapping window |
| `Domain/Query/CascadeMenuBuilder.cs` | `GetNextLevel(parentId)` → `MenuLevel` (Columna + options list) |
| `Domain/Catalog/PartidaConstructibilityResolver.cs` | DFS at startup; `IsConstructible(code)` O(1); `GetPath(code)` returns connection-ID list |
| `Domain/Catalog/PartidaHierarchyResolver.cs` | Longest-prefix match; `Resolve(10-digit-code)` → `(Capitulo, Subcapitulo, Seccion)` |
| `Domain/Catalog/PartidaFilter.cs` | Pure predicate; `ApplyFilter(cap, sub, sec, prefix)` → filtered partida list |
| `Revit/Families/IFamilyGenerator.cs` | Strategy interface + `GeneratorInput` record |
| `Revit/Families/MuroGenerator.cs` | Resolves material + thickness; creates WallType + CompoundStructure + SharedParams |
| `Revit/Families/FamilyGenerationOrchestrator.cs` | Owns both Revit transactions; called via PostExternalEvent |
| `Revit/SharedParameters/CoveninParameters.cs` | 4 stable GUIDs; `EnsureBoundToProject(doc)` idempotent |
| `Revit/SharedParameters/SharedParameterWriter.cs` | Writes the 4 param values on an Element |
| `Revit/Materials/ExtensibleStorageMaterialMappingRepository.cs` | Per-project mapping in ExtensibleStorage; stable Schema GUID |
| `Revit/Context/RevitContextImpl.cs` | Wraps UIApplication; `PostExternalEvent(Action<Document>)` |
| `Ui/AgregarFamilia/AgregarFamiliaViewModel.cs` | All logic for the three-panel window; zero RevitAPI types |
| `Ui/AgregarFamilia/CascadeRowViewModel.cs` | Single dropdown level; `SetSelectedSilently`; `SelectedOptionIsRange`; `RangeInput` |
| `Ui/AgregarFamilia/AgregarFamiliaWindow.xaml` | Three-panel layout; cascade rows; "Seleccionar Partida" button |
| `Ui/Common/FilteredPartidaCollection.cs` | In-memory filter wrapper over `IReadOnlyList<Partida>` |
| `Ui/MaterialMapping/MaterialMappingWindow.xaml` | Mapeo de Materiales window |

---

## Shared parameters — stable GUIDs

**NEVER change these.** Changing them orphans existing `.rvt` files.

| Name | GUID |
|---|---|
| `Capitulo_COVENIN` | `A1B2C3D4-E5F6-7890-ABCD-EF1234567890` |
| `Subcapitulo_COVENIN` | `B2C3D4E5-F6A7-8901-BCDE-F01234567891` |
| `Seccion_COVENIN` | `C3D4E5F6-A7B8-9012-CDEF-012345678912` |
| `Codigo_COVENIN_Completo` | `D4E5F6A7-B8C9-0123-DEFA-123456789012` |

All four are `TypeBinding` (on the WallType, not on instances), placed in `GroupTypeId.IdentityData`.

---

## Material mapping — key inversion pattern

`IMaterialMappingRepository.GetAll()` returns `Dictionary<string revitMaterialName, string coveninValueId>`.

To go from a selected COVENIN `IdValor` → Revit material name:
```csharp
var allMappings = _materialMappingRepo.GetAll();
var entry = allMappings.FirstOrDefault(kvp =>
    string.Equals(kvp.Value, selectedIdValor, StringComparison.OrdinalIgnoreCase));
// entry.Key = Revit material name (or null if unmapped)
```

This inversion is done in both `MuroGenerator` (for element creation) and `AgregarFamiliaViewModel.RefreshRevitMaterials` (for Stage 2 dropdown population).

---

## DAG — empty bridge connections

`Covenin_Conexiones` rows where `CodigoAportado = ""` are **empty bridges** — structurally valid edges that pass through without contributing to the assembled code. They are handled by `EmptyBridgeResolver` and are transparent to `CascadeMenuBuilder` (it never shows them as options; they're intermediate nodes). The key place where empty bridges matter:

- **Cascade seeding**: greedy prefix-walk fails on them; always use `ConstructibilityResolver.GetPath()` for seeding.
- **Code assembly**: `CodigoAportado = ""` contributes nothing to the assembled string — `AssembleCodeFromPath` skips zero-length contributions.

---

## Known non-issues

| Symptom | Explanation |
|---|---|
| PostBuild MSB3021 error when Revit is running | Revit locks the DLL. Close Revit before rebuilding. Not a code error. |
| ~1/3 of E4 partidas shown greyed | Their codes have no valid DAG path (likely typos in the source data). Expected. |
| Only E41x tree nodes are enabled | Only `MuroGenerator` is registered. Correct for MVP. |
| "Sin reglas COVENIN" for non-E4 capítulos | No DAG rows exist for them. Correct; UI degrades gracefully. |

---

## What Phase 5 needs to know

Phase 5 (Asignar Código) reuses the same three-panel window with a `Mode` toggle (`Generate | Assign`). Key additions:

1. **Rename** `AgregarFamiliaWindow` → `PartidaSelectionWindow`; add `Mode` property to the VM.
2. **In Assign mode**: the window accepts a pre-selected Revit element (or shows an in-window element picker). The element's category, dimensions, and materials drive auto-prefill.
3. **Domain/Matching/** (Phase 5.3, not yet built): `CategoryMatcher`, `DimensionalRangeMatcher` use `Valor.NumMin/NumMax/Unidad` for range matching. Material matching delegates to `IMaterialMappingResolver`.
4. **Prefill visual states**: `CascadeRowViewModel` will need a `PrefillState` enum (`AutoFilled | UserOverridden | Undetectable`) in addition to `IsSeeded`. The UI shows highlighted auto-filled rows and greyed "undetectable" prompts.
5. The **existing** `PartidaConstructibilityResolver.GetPath()` is reused verbatim for the Assign backfill flow — no changes needed there.
6. `AsignarCodigoCommand` is a new `IExternalCommand` following the same pattern as `AgregarFamiliaCommand`.

---

## What Phase 6 needs to know

Phase 6 (Codificación Dashboard) is a read-heavy window:

1. `ProjectInventoryReader` (Phase 6.2, not yet built) walks the active document with `FilteredElementCollector`, reads the 4 COVENIN shared params from each element, and returns a grouped summary.
2. The dashboard VM needs to refresh on demand ("Refrescar" button) — no live document-event subscription for MVP.
3. "Generar Schedule" creates a new `ViewSchedule` each time (fire-and-forget; no tracking). See ADR 2026-05-31 — Fire-and-forget schedules.
4. Row click-through opens the Assign mode of `PartidaSelectionWindow` with the element pre-selected.

---

## How to build and test

```powershell
# Rebuild (requires Revit 2026 installed at default path; close Revit first)
dotnet build Ponchevit.slnx

# Run all tests
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj

# Run a single test
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj --filter "FullyQualifiedName~YourTestName"
```

PostBuild copies `Ponchevit.dll` and `manifest/Ponchevit.addin` to `%AppData%\Autodesk\Revit\Addins\2026\`. Smoke-testing in Revit 2026 is the only integration verification path.

**Current test count:** 92 passing, 0 failing. Tests cover Domain + Data only; Revit-layer code is verified by manual smoke test.

---

## Conventions reminder

- **Commits**: Conventional Commits — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Subject in English, imperative. Body references roadmap task IDs.
- **Language**: docs in English; domain terms stay in Spanish (`Capítulo`, `Partida`, `Muro`, `Agregar`, etc.).
- **No new frameworks**: no DI container, no extra NuGet packages without explicit discussion.
- **No git operations**: the user runs all `git commit` / `git push` / `git add` themselves. Never offer or perform git writes.
- **Roadmap**: mark tasks complete with commit SHA. Any plan deviation → same-day entry in `decisions.md`.
