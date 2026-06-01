# Revit Integration Reference

Covers the four RevitAPI-touching pieces Ponchevit owns: shared parameters, per-project material mapping (ExtensibleStorage), the ExternalEvent bridge, and the family generator pattern.

## Shared parameters

### The 4 COVENIN instance parameters

| Name | GUID (source-of-truth in `CoveninParameters.cs`) | Type |
|---|---|---|
| `Capitulo_COVENIN` | `A1B2C3D4-E5F6-7890-ABCD-EF1234567890` | TEXT |
| `Subcapitulo_COVENIN` | `B2C3D4E5-F6A7-8901-BCDE-F01234567891` | TEXT |
| `Seccion_COVENIN` | `C3D4E5F6-A7B8-9012-CDEF-012345678912` | TEXT |
| `Codigo_COVENIN_Completo` | `D4E5F6A7-B8C9-0123-DEFA-123456789012` | TEXT |

**These GUIDs are permanent.** They are declared as `static readonly Guid` constants in `Revit/SharedParameters/CoveninParameters.cs` — the single source of truth. `Resources/SharedParameters.txt` is regenerated from these constants whenever it is missing; never edit the file manually.

Changing any GUID orphans every existing `.rvt` that uses Ponchevit: the old parameters become invisible to the plugin while remaining in the project under their original names.

### Binding

`CoveninParameters.EnsureBoundToProject(Document doc)` is idempotent. It:

1. Regenerates `SharedParameters.txt` beside the DLL if missing.
2. Sets the application's shared-parameter file to that path (restores the original on exit).
3. Gets or creates the definition group `"Ponchevit COVENIN"`.
4. For each of the 4 parameters, creates the `ExternalDefinition` if absent, then binds as `InstanceBinding` to **all model categories** (`CategoryType.Model && AllowsBoundParameters`) if not already bound.
5. All work happens inside a single `Transaction` owned by this method.

Call it at the start of any command that writes COVENIN params to elements.

### Writing parameters

`SharedParameterWriter.Write(element, codigo, capitulo, subcapitulo, seccion, extras?)` uses the caller's transaction. The `extras` parameter is reserved for Post-MVP (see ADR 2026-05-31 — Extras dictionary); passing a non-empty dict throws `NotImplementedException`.

---

## Per-project material mapping — ExtensibleStorage

### Schema

| Field | Value |
|---|---|
| Schema GUID | `E5F6A7B8-C9D0-1234-EF01-234567890123` |
| Schema name | `PonchevitMaterialMapping` |
| Access | Public read + write |
| Payload | Map field `"Entries"`: `IDictionary<string, string>` — Revit material name → Covenin value ID |

The Schema GUID is permanent (same rule as shared-parameter GUIDs). It lives in `Revit/Materials/ExtensibleStorageMaterialMappingRepository.SchemaGuid`.

### How it works

- One `DataStorage` element per document holds the mapping.
- `ExtensibleStorageMaterialMappingRepository.FindStorage()` locates it via `FilteredElementCollector` checking `GetSchemaGuids()`.
- All write operations (`Set`, `Remove`, `Clear`) open their own `Transaction`.
- The mapping travels with the `.rvt` file automatically, including over cloud workshare.

### Payload evolution

If the mapping payload needs extra fields (e.g., `lastEditedBy`), switch the serialized value from a bare string to a versioned JSON object (`{ "version": 2, "value": "...", "editedBy": "..." }`). Do **not** mint a new Schema GUID.

---

## ExternalEvent bridge — IRevitContext

Modeless WPF windows cannot call RevitAPI directly. They use `IRevitContext.PostExternalEvent(Action<Document> work)` to queue document-modifying code onto Revit's main thread.

### Lifecycle

1. `RevitContextImpl` is created in `Services.Build()` (at add-in startup). This registers the `IExternalEventHandler` with Revit immediately.
2. Each `IExternalCommand.Execute` calls `App.Services.RevitContext.Attach(commandData.Application)` to bind the current `UIApplication`.
3. Modeless windows call `PostExternalEvent(doc => { ... })`. Revit calls the handler on its main thread; the delegate runs with a valid `Document`.

### Thread safety

`RevitEventHandler` uses `Interlocked.Exchange` to swap the pending work item. For MVP, one pending item at a time is sufficient.

---

## Family generator pattern

`IFamilyGenerator` is the Strategy for creating code-compliant element instances.

| Property / Method | Purpose |
|---|---|
| `SupportedCategory` | Filters which generator handles a given category selection in the UI. |
| `Generate(Document, GeneratorInput)` | Creates the element. Called inside a caller-supplied transaction. |

`GeneratorInput` carries the assembled `CodigoCovenin`, hierarchy strings, the selected DAG column→value pairs (`CoveninValues`), and Revit-specific placement data (`RevitParameters`).

### Registered generators (MVP)

| Generator | Category |
|---|---|
| `MuroGenerator` | `OST_Walls` |

Register new generators in `Composition/Services.cs` — add to the `generators` array. The UI (Phase 4) looks up the matching generator via `SupportedCategory` when the user clicks "Agregar".

`MuroGenerator.Generate` is stubbed with `NotImplementedException` until Phase 4.7.

---

## Adding a new element type (Post-MVP pattern)

1. Create `Revit/Families/XxxGenerator.cs` implementing `IFamilyGenerator`.
2. Register it in `Services.Build()` alongside `MuroGenerator`.
3. Ensure the left tree in `AgregarFamiliaWindow` enables the corresponding `BuiltInCategory` section (it checks the generator registry).
