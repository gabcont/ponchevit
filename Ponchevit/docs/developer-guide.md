# Ponchevit Developer Guide

A practical, end-to-end introduction to the Ponchevit codebase for a developer who has just been handed the project. By the time you finish reading you should be able to:

- find your way around the repository without grep-fishing,
- understand *why* the architecture looks the way it does,
- run, build, smoke-test, and debug the add-in,
- ship a new architectural-element module (e.g. *Puerta*, *Ventana*, *Piso*) without breaking the rules the existing modules respect.

This guide is the on-ramp; once you've read it, the deep references are:

| You want to know… | Read |
|---|---|
| The big picture and layer boundaries | `architecture.md` |
| The flat catalog + DAG model | `domain-model.md` |
| The SQLite databases and how they load | `data-layer.md` |
| Shared params, ExtensibleStorage, generators | `revit-integration.md` |
| WPF/MVVM conventions and the three-panel window | `ui-patterns.md` |
| Phase status, what's done, what's not | `roadmap.md` |
| Why a thing is the way it is | `decisions.md` (ADRs) |

When this guide and a reference disagree, the reference wins — this guide is a tour, not the spec.

---

## 1. Introduction

### What Ponchevit is

Ponchevit is a Revit 2026 add-in for the Venezuelan **COVENIN E4** construction-norm. Architects and civil engineers use it to attach standardized 10-digit COVENIN codes to elements in a Revit model, and to produce a budgeting-ready report (a native Revit `ViewSchedule`) at the end.

The MVP supports four ribbon commands (Tab "Ponchevit USM" → panel "Acciones"):

| Command | Purpose |
|---|---|
| **Mapeo de Materiales** | Map each Revit project material to a COVENIN material value (per-project, stored inside the .rvt). |
| **Agregar Familia** | Pick a COVENIN code through a three-panel UI and have the plugin create a code-compliant `WallType` with `CompoundStructure` and 4 shared parameters. |
| **Asignar Código** | Tag an existing element with a COVENIN code; auto-prefill what can be inferred (category, dimensions, mapped material) and let the user fill the rest. |
| **Codificación Dashboard** | List placed family types with their codification status, click-through to Asignar, and "Generar Schedule" to materialize a Revit `ViewSchedule`. |

Today only **Muros** are constructible (E41x prefix); doors, windows, floors, etc. work via *Asignar* (codes can be written on existing elements) but cannot yet be *generated* (no `IFamilyGenerator` implementation). Extending the generator catalog is exactly what Section 7 of this guide is about.

### Scope notes — what we deliberately do *not* do

- No DI container. Composition is a hand-rolled root in `Composition/Services.cs`.
- No MVVM framework beyond `CommunityToolkit.Mvvm` *source generators*. No Prism, ReactiveUI, Caliburn.Micro.
- No third-party WPF control library (no MahApps, MaterialDesignInXamlToolkit, DevExpress, Telerik, Syncfusion). Raw WPF only — keeps the addin from clashing with Revit's loaded WPF assemblies.
- No EF Core. Raw `Microsoft.Data.Sqlite` with hand-written `SELECT` queries.
- No async-everywhere. Revit's API is single-threaded; we use `ExternalEvent` for cross-thread work.
- No installer. PostBuild copies the DLL + DBs + manifest to `%AppData%\Autodesk\Revit\Addins\2026\`.
- No CI. Smoke-testing in Revit 2026 is the only integration verification path.

If you find yourself wanting one of these, read the relevant ADR in `decisions.md` first — most have been considered and explicitly rejected for reasons that still apply.

### Project status (as of writing)

Phases 0–6 are implemented and smoke-tested in Revit 2026. The remaining work (Phase 7) is icons, polish, and final-pass docs. The roadmap noted plans to expand the generator catalog to windows and doors; the user has paused that for now, so the patterns in Section 7 are documented but not yet exercised by a second generator.

---

## 2. Architecture

### 2.1 Layered architecture — the hard rule

The codebase is layered by *what each layer is allowed to reference*. Violations break the test project, which is compiled headlessly without RevitAPI.

```
┌──────────────────────────────────────────────────────────────────┐
│  Commands/   thin IExternalCommand entrypoints (may use Revit)   │
│  App.cs      ribbon wiring, OnStartup/OnShutdown (may use Revit) │
├──────────────────────────────────────────────────────────────────┤
│  Ui/         WPF + MVVM (NO direct Revit calls; via IRevitContext)│
├──────────────────────────────────────────────────────────────────┤
│  Revit/      RevitAPI consumers (the ONLY layer besides Commands) │
├──────────────────────────────────────────────────────────────────┤
│  Domain/     pure C# (zero RevitAPI, zero Data dependency on Sqlite)│
│  Data/       interfaces + Sqlite impls (zero RevitAPI)            │
│  Infrastructure/ ILog + FileLog (pure C#)                          │
└──────────────────────────────────────────────────────────────────┘
```

Allowed reference directions (top → bottom):

- `Commands/` → `Ui/`, `Revit/`, `Domain/`, `Data/`, `Composition/`, `Infrastructure/`
- `Ui/`      → `Domain/`, `Data/`, `Revit/Context/`, `Revit/Families/` (interfaces only), `Infrastructure/`
- `Revit/`   → `Domain/`, `Data/`, `Infrastructure/`
- `Domain/`  → `Data/` (interfaces only), `Infrastructure/`
- `Data/`    → `Domain/`, `Infrastructure/`

A simple rule of thumb: if a class lives under `Domain/` or `Data/` and you find yourself reaching for `using Autodesk.Revit.DB;`, **stop** — you're at a layer boundary. The fix is to define an interface in `Data/` (or `Domain/`) and put the Revit-touching implementation under `Revit/`. The canonical example is `IMaterialMappingRepository` (interface in `Data/`, `ExtensibleStorageMaterialMappingRepository` in `Revit/Materials/`).

The same rule keeps `Ui/` clean: `Ui/` never names `Document`, `Transaction`, `ElementId`, `FilteredElementCollector`, `Wall`, `Material`, etc. When it needs work done on Revit's main thread, it calls `IRevitContext.PostExternalEvent(Action<Document>)`. When it needs an existing element by ID, the Revit-layer service exposes a `long`-accepting overload (see `AssignCodeOrchestrator.Assign(Document, long, AssignInput)` and ADR 2026-06-07).

### 2.2 The three data stores

| Store | Where it lives | Shape | Loading |
|---|---|---|---|
| `partidas.db` | `Resources/` SQLite, ~468 KB | Flat catalog: `Capitulos` (10), `Subcapitulos` (46), `Secciones` (190), `Partidas` (2081) | Eager-load all four tables at startup. |
| `covenin.db` | `Resources/` SQLite, ~25 MB | DAG: `Covenin_Columnas` (45), `Covenin_Valores` (379), `Covenin_Conexiones` (~377k) | Columnas + Valores eager; **Conexiones lazy by `Parent_Id`** with in-memory cache. |
| Material mapping | Inside each `.rvt` (Revit `ExtensibleStorage`) | `Dictionary<string,string>` — Revit material name → COVENIN value ID | Read/written via `IMaterialMappingRepository`. |

The split exists because the two SQLite stores have orthogonal lifecycles and shapes — the flat catalog comes from one Python pipeline, the DAG from another. Merging them would couple unrelated schemas. See ADR 2026-05-26 — Two-database strategy.

Two important shape facts you will hit:

- **`Partidas` has no FK to `Secciones`.** The link is computed at catalog load via *longest-prefix match* in `PartidaHierarchyResolver`. See ADR 2026-05-26.
- **Roughly 1/3 of E4 partidas are "unconstructible"** — the DAG has no root-to-leaf path producing their code. They remain visible in the right panel but greyed out. This is a source-data issue, not a code bug.

### 2.3 The composition root

`Composition/Services.cs` is the single file that wires the entire dependency graph. It is instantiated *once* in `App.OnStartup` and consumed everywhere via `App.Services`. There is no DI container — if you add a new service, you add a field, a constructor parameter, and a line in `Services.Build()`.

```csharp
// Composition/Services.cs (abbreviated)
public sealed class Services
{
    public ILog Log { get; }
    public RevitContextImpl RevitContext { get; }
    public IPartidasRepository PartidasRepository { get; }
    public ICoveninRulesRepository CoveninRulesRepository { get; }
    public PartidaCatalog PartidaCatalog { get; }
    public PartidaConstructibilityResolver ConstructibilityResolver { get; }
    public IFamilyGenerator[] FamilyGenerators { get; }
    public IElementRecognizer[] ElementRecognizers { get; }
    // …

    public static Services Build() { /* construct everything top-down */ }
}
```

A consequence of the manual root is that adding a service is *non-magical* — there is exactly one place to look. The flip side is that you cannot forget to register it; it will refuse to compile.

### 2.4 The ExternalEvent bridge

Revit's API is single-threaded. A modeless WPF window runs on the WPF UI thread, not Revit's main thread, so it cannot call `RevitAPI` directly. The bridge is `IRevitContext.PostExternalEvent(Action<Document>)`:

```csharp
// Ui/ — VM posts a write back to Revit
_revitContext.PostExternalEvent(doc =>
{
    using var t = new Transaction(doc, "My op");
    t.Start();
    // RevitAPI calls run here, on Revit's main thread, with a valid Document
    t.Commit();
});
```

`RevitContextImpl` registers an `IExternalEventHandler` once at startup. The handler swaps a pending `Action<Document>` via `Interlocked.Exchange` — at most one pending item at a time is enough for MVP. The orchestrator pattern (see Section 2.5) keeps the delegate body small: the VM never opens transactions; it posts a delegate that invokes an orchestrator that owns the transaction.

### 2.5 Orchestrators

Two `Revit/` classes own the "do real work in Revit" transactions: `FamilyGenerationOrchestrator` and `AssignCodeOrchestrator`. They exist for one reason — to keep `Ui/` clean. The VM cannot mention `Transaction` (RevitAPI type), so the VM calls `PostExternalEvent(doc => orchestrator.Generate(doc, generator, input))` and the orchestrator owns:

1. `CoveninParameters.EnsureBoundToProject(doc)` (Transaction 1, idempotent, internal).
2. `new Transaction(doc, "…").Start()` (Transaction 2, wrapping the generator/writer call).

See ADR 2026-06-06 — Transaction-orchestration location and ADR 2026-06-07 — long-accepting overloads.

### 2.6 The three storage rules

These three rules anchor everything else:

1. **The .rvt is the only place state lives.** No sidecar files, no per-user state on disk beyond `%AppData%\Ponchevit\log.txt`. Shared parameters, material mapping, schedules — all inside the .rvt.
2. **GUIDs are permanent.** The 4 COVENIN shared-parameter GUIDs (in `CoveninParameters.cs`) and the ExtensibleStorage Schema GUID (in `ExtensibleStorageMaterialMappingRepository.cs`) **must never change**. Changing one orphans every existing .rvt that uses Ponchevit. See ADR 2026-05-31 — GUID source-of-truth.
3. **Payload evolution stays inside the schema.** If the material-mapping payload needs new fields, switch the serialized value to a versioned JSON object. Do *not* mint a new Schema GUID.

---

## 3. Repository structure

```text
Ponchevit.slnx                       solution root
Ponchevit/                           the only buildable C# project
│
├── App.cs                           IExternalApplication; ribbon wiring; OnStartup
├── Ponchevit.csproj                 net8.0-windows; UseWPF=true; PostBuild copies to Addins\2026\
│
├── Commands/                        thin IExternalCommand entrypoints (one per ribbon button)
│   ├── MapeoMaterialesCommand.cs
│   ├── AgregarFamiliaCommand.cs
│   ├── AsignarCodigoCommand.cs
│   └── CodificacionDashboardCommand.cs
│
├── Composition/
│   └── Services.cs                  manual composition root, the dependency graph
│
├── Domain/                          PURE C# — zero RevitAPI refs
│   ├── Model/                       Capitulo, Subcapitulo, Seccion, Partida, Columna, Valor, Conexion, CodigoCovenin
│   ├── Graph/                       EmptyBridgeResolver, CodeAssembler (root→leaf concat + 10-digit firewall)
│   ├── Query/                       PrefixPathQuery, CascadeMenuBuilder (drives central panel)
│   ├── Catalog/                     PartidaCatalog, PartidaHierarchyResolver, PartidaFilter,
│   │                                PartidaConstructibilityResolver
│   ├── Aliases/                     IAliasResolver, IdentityAliasResolver (passthrough for MVP)
│   ├── Materials/                   IMaterialMappingResolver, SubstringSuggester
│   ├── Matching/                    ElementTopology, PrefillResult (recognizer inputs/outputs)
│   └── Codificacion/                CodificacionSummary (record)
│
├── Data/                            PURE C# — zero RevitAPI refs
│   ├── IPartidasRepository.cs
│   ├── ICoveninRulesRepository.cs
│   ├── IMaterialMappingRepository.cs
│   └── Sqlite/                      SqlitePartidasRepository, SqliteCoveninRulesRepository, ConnectionFactory
│
├── Revit/                           the only place besides Commands that may use RevitAPI
│   ├── Context/                     IRevitContext, RevitContextImpl (ExternalEvent registration)
│   ├── SharedParameters/            CoveninParameters (4 stable GUIDs + EnsureBoundToProject)
│   ├── SharedParameterWriter.cs     writes the 4 params on an Element
│   ├── ElementTopologyReader.cs     extracts category/layers/dimensions into a pure ElementTopology
│   ├── ProjectMaterialQuery.cs      IProjectMaterialQuery — FilteredElementCollector over Material
│   ├── Materials/                   ExtensibleStorageMaterialMappingRepository (Schema GUID lives here)
│   ├── Codificacion/                ProjectInventoryReader, CodificacionScheduleBuilder
│   └── Families/                    IFamilyGenerator, MuroGenerator,
│                                    IElementRecognizer, MuroRecognizer,
│                                    FamilyGenerationOrchestrator, AssignCodeOrchestrator, AssignInput
│
├── Ui/                              WPF (XAML + .cs); zero direct RevitAPI calls; uses IRevitContext
│   ├── Common/                      FilteredPartidaCollection (in-memory predicate over partidas)
│   ├── MaterialMapping/             MaterialMappingWindow + ViewModel (Phase 3)
│   ├── PartidaSelection/            PartidaSelectionWindow + ViewModel (Agregar + Asignar; Mode toggle)
│   │                                CascadeRowViewModel, TreeNodeViewModel, PartidaDisplayItem,
│   │                                PrefillReportLine, WindowMode, Converters
│   └── Codificacion/                CodificacionDashboardWindow + ViewModel + CodificacionRowViewModel
│
├── Infrastructure/
│   ├── ILog.cs                      Info / Warn / Error(message, ex?)
│   └── FileLog.cs                   → %AppData%\Ponchevit\log.txt
│
├── Resources/
│   ├── partidas.db                  shipped + copied beside DLL by PostBuild
│   ├── covenin.db                   shipped + copied beside DLL by PostBuild
│   ├── SharedParameters.txt         regenerated from CoveninParameters constants if missing
│   └── icons/                       (Phase 7) ribbon icons
│
├── manifest/
│   └── Ponchevit.addin              AddInId GUID is permanent (1bfbb086-…); copied to Addins\2026\
│
└── docs/                            authoritative design specs — read before changing behavior
    ├── architecture.md              layer + composition + storage
    ├── domain-model.md              flat catalog + DAG + CodigoCovenin
    ├── data-layer.md                SQLite + ConnectionFactory + loading strategies
    ├── revit-integration.md         shared params, ExtensibleStorage, ExternalEvent, IFamilyGenerator
    ├── ui-patterns.md               modeless windows, MVVM, two-stage material control, etc.
    ├── codificacion-dashboard.md    (TODO — Phase 7.4)
    ├── roadmap.md                   single source of truth for phase status
    ├── decisions.md                 ADRs
    ├── handoff.md                   end-to-end session-bootstrap doc
    └── developer-guide.md           this file

Ponchevit.Tests/                     xUnit (net8.0-windows); references main project; covers Domain + Data
│
├── SmokeTests.cs
├── Composition/                     ServicesHierarchyResolverTests
├── Domain/Graph/ Query/ Catalog/ Materials/  ~80 unit tests
├── Data/                            RepositoryTests (in-memory SQLite fixtures)
├── Revit/Families/                  MuroGeneratorSupportsTests (pure logic only — no live Document)
└── Ui/Common/                       FilteredPartidaCollectionTests

Partidas/, Tablas/                   Python data-extraction pipelines that produce partidas.db & covenin.db.
                                     NOT part of the .NET solution. Treat as upstream black boxes.
```

### What goes where — quick decision tree

When you don't know which folder a new file should go in, walk this tree top-down:

1. Does it produce a ribbon button? → `Commands/`
2. Does it touch RevitAPI? → `Revit/` (or `Commands/`)
3. Is it a XAML window / VM? → `Ui/<Feature>/`
4. Does it talk to SQLite or ExtensibleStorage? → interface in `Data/`, impl in `Data/Sqlite/` or `Revit/<Feature>/`
5. Is it pure logic — code assembly, prefix-matching, predicates, value types? → `Domain/`
6. Is it a logger or persistence-free utility? → `Infrastructure/`

If it touches two layers, you've probably described two files.

---

## 4. Standards and conventions

### 4.1 Language

- Documentation prose: **English**.
- Identifiers for domain concepts: **Spanish** — `Capitulo`, `Subcapitulo`, `Seccion`, `Partida`, `Muro`, `Agregar`, `Asignar`, `Codificacion`, `MapeoMateriales`, etc.
- Code comments: English. Spanish only when quoting a domain term or a user-facing string.

This split is deliberate: domain terms come from the COVENIN norm and shouldn't be translated. See ADR 2026-05-26 — Docs in English, domain terms in Spanish.

### 4.2 Commits

**Conventional Commits.** Subject in English, imperative. Reference roadmap task IDs in the body.

```
feat(domain): add CodeAssembler (roadmap 1.3)
fix(ui): stop greedy prefix-walk during cascade seeding (ADR 2026-06-06)
docs(architecture): write developer guide
refactor(revit): extract FamilyGenerationOrchestrator
test(data): cover ConnectionFactory schema validation
```

Allowed prefixes: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`, `build`, `ci`. See ADR 2026-05-26 — Conventional Commits.

When a task ships, **update the corresponding checkbox in `roadmap.md` with the commit SHA** before moving on. Any deviation from the plan goes into `decisions.md` the same day as a new ADR entry.

### 4.3 C# style

- TFM: `net8.0-windows`. `Nullable` enabled. `ImplicitUsings` enabled. `UseWPF` enabled.
- File-scoped namespaces (`namespace Foo.Bar;`).
- `record` / `record class` for immutable models (`Capitulo`, `Valor`, `Conexion`, `GeneratorInput`, `PrefillResult`, etc.).
- `sealed class` for services unless inheritance is required.
- Prefer `IReadOnlyList<T>` / `IReadOnlyDictionary<TKey,TValue>` in public surfaces.
- Constructor injection. Throw `ArgumentNullException` at the top of the constructor for required dependencies.
- One class per file. Internal helper records can sit alongside their owning class if they make no sense elsewhere.
- `.editorconfig` covers the defaults.

### 4.4 Comments

Default to **no comments**. A short one-liner is fine when *why* is non-obvious — a constraint, a subtle invariant, a workaround for a Revit-API quirk. Pointers into ADRs are gold (`// See ADR 2026-06-06 — Cascade seeding uses GetPath`). Don't write multi-paragraph docstrings. Don't restate what the code does.

XML doc-comments (`/// <summary>…</summary>`) are *encouraged* on public types in `Domain/`, `Data/`, and `Revit/` — they document the contract for the next developer. They are not required on private helpers or `Ui/` VMs (the XAML is the contract there).

### 4.5 Logging

`ILog` has three levels: `Info`, `Warn`, `Error(message, ex?)`. Default to `Info` on successful side-effecting operations (transaction commit, schedule created), `Warn` on graceful degradations (anomaly excluded from catalog, recognizer couldn't resolve structural path), `Error` on caught exceptions. All log lines land in `%AppData%\Ponchevit\log.txt` via `FileLog`.

Never `Console.WriteLine`. Never throw away an exception — catch it in the Command, log it, set the user-facing `message`, return `Result.Failed`.

### 4.6 MVVM conventions

ViewModels inherit from `CommunityToolkit.Mvvm.ComponentModel.ObservableObject`. Properties use source-generator attributes:

```csharp
public partial class MyViewModel : ObservableObject
{
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsValid))]
    private string _input = string.Empty;

    public bool IsValid => !string.IsNullOrWhiteSpace(_input);

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SaveCommand))]
    private bool _canSave;

    [RelayCommand(CanExecute = nameof(CanSave))]
    private void Save() { /* … */ }

    [RelayCommand]
    private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);

    public event EventHandler? CloseRequested;
}
```

The VM never holds a `Window` reference. It raises `CloseRequested`; the window subscribes in its constructor and calls `Close()`. Never call `Close()` from the VM.

### 4.7 Windows are modeless

Use `window.Show()`, never `window.ShowDialog()`. Revit stays responsive while the window is open. The `IExternalCommand.Execute` returns `Result.Succeeded` immediately after `Show()` — the window owns its own lifecycle from there.

A consequence: VMs **cannot** call RevitAPI directly. They cross the thread boundary via `IRevitContext.PostExternalEvent`. Picking elements (`UIDocument.Selection.PickObject`) is not possible while a modeless window is open — that's why *Asignar Código* requires the element to be pre-selected in Revit before clicking the ribbon button, or passed in from the Dashboard. See ADR 2026-06-06 — No in-window element picker.

### 4.8 Spelling: IsGenerable vs CanBeConstructed

Two superficially-similar predicates exist; keep them straight (ADR 2026-06-06 — Naming):

- `IFamilyGenerator.IsGenerable(string? codigoPrefix)` — a *generator module* exists for this code prefix. Used to enable left-tree sections in Generate mode.
- `PartidaConstructibilityResolver.CanBeConstructed(Partida)` — at least one root-to-leaf path in the DAG produces this partida's code. Used to grey out right-panel rows in Generate mode.

A partida can be `CanBeConstructed` but not `IsGenerable` (DAG covers it but we haven't shipped the generator); the reverse is not currently possible but the names keep both axes legible. In Assign mode, *neither* check gates partida selection.

---

## 5. Build, run, and test

### 5.1 Prerequisites

- Windows 10/11.
- Revit 2026 installed at the default path (`C:\Program Files\Autodesk\Revit 2026\`). The `.csproj` references RevitAPI/RevitAPIUI by relative HintPath from there.
- .NET 8 SDK.
- Visual Studio 2022+ or `dotnet` CLI is enough.

### 5.2 Build

```powershell
# From the repo root:
dotnet build Ponchevit.slnx
```

PostBuild copies the following to `%AppData%\Autodesk\Revit\Addins\2026\`:

- `Ponchevit.dll` and its NuGet dependency DLLs (`CommunityToolkit.Mvvm`, `Microsoft.Data.Sqlite`, …).
- `e_sqlite3.dll` (the native SQLite binary for win-x64 — required for `Microsoft.Data.Sqlite`'s P/Invoke).
- `Ponchevit.deps.json` (so the .NET runtime resolves assembly versions inside Revit).
- `manifest/Ponchevit.addin`.
- `partidas.db` and `covenin.db`.

**Common build error:** MSB3021 — the linker can't overwrite `Ponchevit.dll`. Revit is running and has the DLL locked. Close Revit and rebuild.

### 5.3 Run

1. Build successfully (see above).
2. Launch Revit 2026. The "Ponchevit USM" tab appears at the top.
3. Open or create a project. The four ribbon buttons are under panel "Acciones".

To **debug**, configure Visual Studio to launch `C:\Program Files\Autodesk\Revit 2026\Revit.exe` and attach to the process; breakpoints in commands and orchestrators hit on click.

### 5.4 Tests

The headless test project covers `Domain/` and `Data/` only. Revit-touching code is verified by manual smoke-test.

```powershell
# Run all tests
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj

# Run a single test by name (FullyQualifiedName supports a substring filter)
dotnet test Ponchevit.Tests/Ponchevit.Tests.csproj --filter "FullyQualifiedName~CodeAssembler"
```

Test fixtures use **in-memory SQLite databases** (one per schema). See `Ponchevit.Tests/Data/RepositoryTests.cs` for the pattern: create a `SqliteConnection(":memory:")`, run the `CREATE TABLE` + `INSERT` script, hand the connection to the repository.

### 5.5 Logs

When in doubt, check `%AppData%\Ponchevit\log.txt`. Catalog anomalies, recognizer warnings, command-level exceptions, and orchestrator commits all land there.

---

## 6. End-to-end flows — read these once to understand the codebase

### 6.1 Startup

```
Revit launches → loads Ponchevit.addin → instantiates Ponchevit.App
    App.OnStartup(UIControlledApplication)
        ├── Services.Build()
        │     ├── new FileLog()
        │     ├── new RevitContextImpl()    [registers ExternalEvent NOW, before any command]
        │     ├── new ConnectionFactory()
        │     ├── new SqlitePartidasRepository(connFactory)            [eager-loads 4 tables]
        │     ├── new SqliteCoveninRulesRepository(connFactory)        [eager: Columnas, Valores]
        │     ├── new PartidaCatalog(repo, log)                        [resolves hierarchy, drops anomalies]
        │     ├── new PartidaConstructibilityResolver(rulesRepo, catalog.GetPartidas())
        │     │       [one-time DFS over DAG with prefix-pruning; builds HashSet<string> + path map]
        │     ├── new IdentityAliasResolver()
        │     ├── new ExtensibleStorageMaterialMappingRepository(() => revitContext.ActiveUiDocument.Document)
        │     ├── new MaterialMappingResolver(mapRepo)
        │     ├── new IFamilyGenerator[]   { new MuroGenerator(mapRepo) }
        │     ├── new IElementRecognizer[] { new MuroRecognizer(rulesRepo, log) }
        │     ├── new FamilyGenerationOrchestrator(log)
        │     ├── new AssignCodeOrchestrator(log, aliasResolver)
        │     ├── new ProjectMaterialQuery(revitContext)
        │     ├── new PartidaHierarchyResolver(…)
        │     ├── new ProjectInventoryReader()
        │     └── new CodificacionScheduleBuilder(log)
        ├── CreateRibbonTab("Ponchevit USM")  [ignore if exists]
        ├── CreateRibbonPanel(tab, "Acciones")
        └── Add four PushButtons → MapeoMateriales, AgregarFamilia, AsignarCodigo, CodificacionDashboard
```

`Services.Build()` is invoked exactly once, and the resulting `Services` instance lives in `App.Services` for the rest of the Revit session.

### 6.2 Command click — generic pattern

Every `IExternalCommand.Execute` follows the same shape:

```csharp
[Transaction(TransactionMode.Manual)]   // or ReadOnly for query-only commands
public class XxxCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);   // bind current UIApplication

            // (optional) read pre-selection or pre-collect project materials

            var vm = new XxxViewModel(/* services */);
            var window = new XxxWindow(vm);
            window.Show();                                            // modeless

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("Xxx failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }
}
```

The command never opens a Revit `Transaction` itself — that's the orchestrator's job, executed later under `PostExternalEvent`. The `TransactionMode.Manual` attribute is required *for the command lifecycle*, not because the command writes during `Execute`.

### 6.3 Agregar Familia — full data flow

This is the most complex flow; once you understand it, the rest follow the same shape.

```
User selects tree node "Muros E411" in PartidaSelectionWindow (Generate mode)
    → PartidaSelectionViewModel.SelectTreeNodeCommand(node)
    → FilteredPartidaCollection.ApplyFilter(capCod, subCod, secCod)
    → SeedCascadeFromTreeNode(node)
        → ConstructibilityResolver.GetPath(seedPartida.CodigoPartida)  [pre-computed list of IdConexion]
        → SeedRowsFromPath(nodeCode, fullPath)                          [adds locked seeded rows]
        → AppendNextCascadeLevel(lastConnId)                            [first user-editable row]
    → UpdateCodeDisplay()                                               [CurrentCodeDisplay = "E411XXXXXX"]
    → RebuildDisplayItems()                                             [right panel narrows]

User changes a cascade dropdown (e.g. selects a MATERIAL value)
    → OnCascadeSelectionChanged(changedRow)
    → Rebuild _selectedConnectionPath up to changedRow + new selection
    → Truncate rows below changedRow
    → If material column → RefreshRevitMaterials(row)                   [Stage 2 dropdown]
    → AppendNextCascadeLevel(changedRow.SelectedOption.IdConexion)
    → UpdateCodeDisplay() + UpdateRightPanelFromPath() + UpdateCanPerformAction()

User clicks "Seleccionar Partida" with a constructible partida highlighted
    → ConfirmPartidaCommand
    → BackfillCascadeFromPath(GetPath(selectedPartida.CodigoPartida))
        [for each connId: build CascadeRowViewModel, SetSelectedSilently(option)]
    → UpdateCodeDisplay() → "E4110705FF" (full 10 digits)
    → CanPerformAction = true

User clicks "Agregar"
    → PerformActionCommand
    → BuildGeneratorInput()
        SelectedValores: IdColumna → Valor (full object, has NumMin/NumMax/Unidad)
        NumericValues:   IdColumna → double in metres (range rows with typed value only)
        Descripcion:     PartidaCatalog lookup on assembled code
        Codigo:          new CodigoCovenin(assembled10digits)
        Capitulo/Subcapitulo/Seccion: HierarchyResolver.Resolve(assembled10digits) titles
    → _revitContext.PostExternalEvent(doc =>
          orchestrator.Generate(doc, matchingGenerator, input))

        [Revit main thread — FamilyGenerationOrchestrator.Generate]
        Transaction 1: CoveninParameters.EnsureBoundToProject(doc)
            - regenerates SharedParameters.txt from constants if missing
            - sets app.SharedParametersFilename
            - creates DefinitionGroup "Ponchevit COVENIN" if missing
            - for each of 4 params: create ExternalDefinition if missing; bind as TypeBinding to OST_Model* if not bound
            - commits, restores previous SharedParametersFilename
        Transaction 2: MuroGenerator.Generate(doc, input)
            Step 1 — Resolve Revit material
                allMappings = materialMappingRepo.GetAll()              [Dict<revitName, coveninValueId>]
                Find mapping entry where .Value == any selected IdValor [invert lookup]
                FindRevitMaterialId(doc, mappingEntry.Key)              [FilteredElementCollector over Material]
            Step 2 — Resolve thickness
                if NumericValues has it → use it; convert to feet via UnitUtils
                else find first SelectedValor with NumMin != null and not a material → use NumMin
                fallback 10 cm
            Step 3 — Resolve WallType
                check for existing WallType named "COVENIN {code} — {desc}"; reuse if present
                else duplicate first WallKind.Basic WallType under that name
            Step 4 — Build CompoundStructure with a single layer (resolved material + thickness)
                CompoundStructure.CreateSimpleCompoundStructure(layers)
                newWallType.SetCompoundStructure(cs)
            Step 5 — SharedParameterWriter.Write(newWallType, codigo, capitulo, subcapitulo, seccion)
                LookupParameter(name).Set(value) for each of the 4 params on the WallType
        Transaction 2 commits

    → Dispatcher.Invoke: StatusMessage = "Familia creada correctamente." (green)
        on exception: StatusMessage = "Error: {ex.Message}" (red); orchestrator logs the exception
```

Key things to internalize from this flow:

- `MuroGenerator` creates a **WallType**, not a `Wall` instance. The 4 shared params are `TypeBinding` and sit in the "Identity Data" section of the Properties panel for the new type. See ADR 2026-06-06 — MuroGenerator creates WallType and ADR 2026-06-06 — TypeBinding and GroupTypeId.IdentityData.
- The two transactions are *separate*. `EnsureBoundToProject` commits before the creation transaction so Revit sees the freshly-bound params during `SharedParameterWriter.Write`.
- "Empty bridges" (`CodigoAportado = ""`) are valid DAG edges that contribute nothing. `CodeAssembler` skips zero-length contributions; `CascadeMenuBuilder` never surfaces them as options (they are intermediate). The cascade seeder uses the pre-computed `GetPath()` precisely because a greedy prefix-walk breaks on empty bridges. See ADR 2026-06-06.

### 6.4 Asignar Código — what differs

Same window, `Mode = Assign` (see `Ui/PartidaSelection/WindowMode.cs`). The behavioral matrix lives in roadmap §Phase 5; the highlights:

- The window receives a `TargetElement` (either pre-selected in Revit, or passed from the Dashboard's row button). No in-window picker.
- All left-tree sections are enabled — Assign works on any element, not only constructible ones.
- All right-panel partidas are selectable (no grey-out).
- The "Reconocer" button calls `services.RecognizeTopology(topology)` → routes to the first `IElementRecognizer.CanRecognize(topology)` and returns a `PrefillResult` per `IdColumna` with state `AutoFilled` | `Ambiguous` | `Undetectable`. Auto-filled rows render highlighted; undetectable rows show a grey prompt. Qualitative columns are returned as `Undetectable` *by design* — no fake confidence (ADR 2026-05-31 — Reconocer dissolved into Asignar).
- The action button is "Asignar" instead of "Agregar". `assignAction` (built in the Command and captured by the VM) calls `AssignCodeOrchestrator.Assign(doc, elementIdLong, AssignInput)` under `PostExternalEvent`. The orchestrator runs `aliasResolver.Resolve` on the incoming code, opens its own transaction, finds the element's *type* (COVENIN params are type-bound), and writes the 4 params.

### 6.5 Codificación Dashboard

```
User clicks "Codificación Dashboard"
    → CodificacionDashboardCommand.Execute
    → services.ProjectInventoryReader.Read(doc)
        FilteredElementCollector over the 10 supported BuiltInCategories
        for each category: group placed instances by GetTypeId().Value
        per group: read CodigoCompletoName param, compute quantity (HOST_AREA_COMPUTED → m²; else count)
        emit CodificacionSummary records, ordered (codified first, then by family/type name)
    → ProjectMaterialQuery.GetProjectMaterials()  [needed because dashboard row → Asignar reuses the window]
    → new CodificacionDashboardViewModel(…)
    → window.Show()

User clicks "Asignar código" on a row
    → opens PartidaSelectionWindow in Assign mode with that family type's SampleInstanceId pre-filled
    → flow continues as in §6.4

User clicks "Generar Schedule"
    → PostExternalEvent → CodificacionScheduleBuilder.Build(doc)
        - EnsureBoundToProject
        - Transaction("Generar Schedule COVENIN")
        - ViewSchedule.CreateSchedule(doc, ElementId.InvalidElementId)         [multi-category]
        - Name = "COVENIN - Codificación 2026-06-24 14-30"
        - AddField for each of the 4 COVENIN shared params (located via stable GUID)
        - AddField Count
        - Group by Codigo, IsItemized = false  [one row per unique code]
        - commit
    → fire-and-forget: plugin never references the schedule again
```

See ADR 2026-05-31 — Fire-and-forget schedules.

### 6.6 Mapeo de Materiales

The simplest flow.

```
User clicks "Mapeo de Materiales"
    → MapeoMaterialesCommand.Execute → window.Show()
    → MaterialMappingViewModel surfaces a 3-column grid:
        Revit material name | Substring suggestion | Covenin material dropdown
    → Suggestion column = SubstringSuggester.Suggest(name, coveninMaterialValues)
        substring + 4+ char word-overlap; pure logic in Domain/Materials
User accepts/overrides each row, clicks Save
    → For each changed row: PostExternalEvent → IMaterialMappingRepository.Set(revitName, coveninId)
        → ExtensibleStorageMaterialMappingRepository.Save()
            - GetOrCreateSchema()  [Schema.Lookup or SchemaBuilder.Finish]
            - FindStorage          [FilteredElementCollector → DataStorage with valid Entity for our Schema]
            - new Transaction("Save Ponchevit material mapping")
            - if storage == null: storage = DataStorage.Create(doc)
            - entity.Set<IDictionary<string,string>>("Entries", dict)
            - storage.SetEntity(entity)
            - commit
```

After save, the data lives inside the .rvt. It survives save/reload and travels over cloud workshare. There is no sidecar.

---

## 7. Adding a new element-type module — the recipe

This is the section the user explicitly asked for. Suppose you want to ship `PuertaGenerator` (doors) so users can *Agregar* a code-compliant door, *Asignar* a code to an existing door with auto-prefill, and see doors in the dashboard.

The recipe follows the existing `Muro` patterns 1:1. There are five surfaces to touch — generator (creation), recognizer (recognition), composition root, generator-enabling logic in the VM, and (if needed) the dashboard's category descriptor.

### 7.1 Decide what "Generate" actually means for your element

For Muros, "Generate" means *create a `WallType` with a `CompoundStructure`* — not a wall instance. Geometry placement (start point, end point, level) is out of scope. The same principle applies to all generators: produce the **family type** asset in the project library, not an instance.

For an element where a code-driven `CompoundStructure` is meaningless (doors, windows, generic models with rich parametric content), the strategy in the ADRs is to **ship a base RFA** in `Resources/Families/` and have the generator load it and configure parameters/type-properties on the resulting `FamilySymbol`. See ADR 2026-06-06 — Base RFA strategy: ship in Resources/Families/.

Decide once, write a sentence in `decisions.md` if it's a non-obvious choice, then continue.

### 7.2 Implement `IFamilyGenerator`

Create `Revit/Families/PuertaGenerator.cs`:

```csharp
using System;
using Autodesk.Revit.DB;
using Ponchevit.Data;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Families;

public sealed class PuertaGenerator : IFamilyGenerator
{
    private readonly IMaterialMappingRepository _materialMappingRepo;

    public PuertaGenerator(IMaterialMappingRepository materialMappingRepo)
    {
        _materialMappingRepo = materialMappingRepo
            ?? throw new ArgumentNullException(nameof(materialMappingRepo));
    }

    public BuiltInCategory SupportedCategory => BuiltInCategory.OST_Doors;

    public bool IsGenerable(string? codigoPrefix)
        => codigoPrefix?.StartsWith("E42", StringComparison.OrdinalIgnoreCase) == true;
        //                                ^^^^^ — whatever the COVENIN prefix is for doors

    public void Generate(Document doc, GeneratorInput input)
    {
        // The transaction is already open (FamilyGenerationOrchestrator owns it).
        //
        // For doors, the cheapest first cut is:
        //   1. Load a base RFA from %AppData%\Autodesk\Revit\Addins\2026\Families\Puerta-Base.rfa
        //      via doc.LoadFamily.
        //   2. Find the first FamilySymbol in that family (or duplicate one with a COVENIN-named type).
        //   3. Configure type-level properties from input.SelectedValores / input.NumericValues
        //      (height/width, finish, material — mediated by IMaterialMappingRepository like MuroGenerator).
        //   4. SharedParameterWriter.Write(typeSymbol, input.Codigo, capitulo, subcapitulo, seccion).
        throw new NotImplementedException("PuertaGenerator: implement following MuroGenerator's shape.");
    }
}
```

Notes:

- `SupportedCategory` is the BuiltInCategory you want to be enabled in the left tree (Generate mode).
- `IsGenerable(prefix)` returns true when the COVENIN code starts with whatever your element's subcapítulo prefix is. Keep this method *cheap* — it gets called a lot during tree-enable computation.
- `Generate` runs inside a Revit transaction already opened by `FamilyGenerationOrchestrator`. **Do not** open another transaction. Throw exceptions; the orchestrator catches them and the VM surfaces them as the red status line.
- Convert all dimensions to feet (Revit internal units) via `UnitUtils.ConvertToInternalUnits(value, UnitTypeId.Meters)` (or cm/mm).
- Always end with `SharedParameterWriter.Write` so the 4 COVENIN params are populated. The orchestrator already called `EnsureBoundToProject` for you.

If you ship a base RFA, put it under `Resources/Families/Puerta-Base.rfa`, add it to `Ponchevit.csproj` with `CopyToOutputDirectory="PreserveNewest"`, and copy it via a new `Copy` step in `PostBuild` to `%AppData%\Autodesk\Revit\Addins\2026\Families\`.

### 7.3 Implement `IElementRecognizer`

Create `Revit/Families/PuertaRecognizer.cs`. Use `MuroRecognizer` as the reference. The shape:

```csharp
public sealed class PuertaRecognizer : IElementRecognizer
{
    public BuiltInCategory SupportedCategory => BuiltInCategory.OST_Doors;

    public bool CanRecognize(ElementTopology topology)
        => topology?.BuiltInCategoryId == (int)BuiltInCategory.OST_Doors;

    public PrefillResult Recognize(ElementTopology topology)
    {
        // 1. Resolve the structural path for doors (CAPITULO + SUBCAPITULO + ACTIVIDAD + UN)
        //    via _rulesRepo.GetConexionesByValorId("VAL_<known door sección>")
        //    and walking parents/children.
        // 2. For each known COL_ in the door rules:
        //    - if quantitative (dimension, mapped material): try to recognize from topology
        //    - if qualitative: return Undetectable
        // 3. Return new PrefillResult(entries).
        //
        // Wrap the whole thing in try/catch and return PrefillResult.Empty on failure.
        // Never throw from a recognizer.
        throw new NotImplementedException("PuertaRecognizer: follow MuroRecognizer's shape.");
    }
}
```

The "Recognize" contract is clear about what's *honest* — qualitative columns must be `Undetectable`, ambiguous matches must be `Ambiguous`. Don't fabricate auto-fill values you can't justify from the topology. The UI surfaces the per-column state to the user; the right answer is "I don't know" when you don't.

If your element's geometry differs from what `ElementTopologyReader` extracts today (it currently knows about `Wall.CompoundStructure`, wall height, wall thickness), extend `ElementTopologyReader.ReadLayers` / `ReadDimensions` to add the parameters you need. Keep adding to the `Dimensions` `IReadOnlyDictionary<string, double>` (in feet); the recognizer pulls by name.

### 7.4 Register the module in `Services.Build()`

`Composition/Services.cs`:

```csharp
var generators  = new IFamilyGenerator[]   { new MuroGenerator(materialMappingRepo),
                                             new PuertaGenerator(materialMappingRepo) };
var recognizers = new IElementRecognizer[] { new MuroRecognizer(coveninRepo, log),
                                             new PuertaRecognizer(coveninRepo, log) };
```

That's it for composition. The arrays are iterated by:

- `Services.RecognizeTopology` — routes a topology to the first recognizer that `CanRecognize`.
- `Services.CanRecognizeTopology` — returns true if any recognizer claims the topology.
- The VM's `BuildTree`/`IsSeccionEnabled` — iterates `_generators` to compute which left-tree sections are enabled.

You do *not* need a switch statement; the strategy is dispatched by `BuiltInCategory` + `IsGenerable(prefix)`.

### 7.5 Wire the left-tree enabling rule

`Ui/PartidaSelection/PartidaSelectionViewModel.BuildTree` enables Sección nodes whose code prefix matches a registered generator. Today, that boils down to a single rule mapping `"E41"` → `OST_Walls`. For Puertas you need the equivalent mapping for the doors' COVENIN subcapítulo prefix.

Look at `IsSeccionEnabled` in `PartidaSelectionViewModel`. Add the prefix → category mapping; pass `_generators` so the same generator-array contract is honored. The XAML's `DataTrigger` on `IsEnabled` greys out non-enabled nodes automatically.

### 7.6 Dashboard category descriptor (if not already present)

`ProjectInventoryReader.SupportedCategories` already lists the common categories (`OST_Walls`, `OST_Floors`, `OST_Ceilings`, `OST_Roofs`, `OST_Doors`, `OST_Windows`, `OST_Columns`, `OST_StructuralFraming`, `OST_Stairs`, `OST_GenericModel`). If your element falls outside that list, add a tuple `(BuiltInCategory, BuiltInParameter? areaParam, string displayName)`:

```csharp
(BuiltInCategory.OST_GenericModel, null, "Modelo genérico"),
```

`areaParam` is `HOST_AREA_COMPUTED` for things that should be reported in m², `null` for things that should be counted (pieces). See ADR 2026-06-06 — Quantity reporting uses native Revit schedule fields.

### 7.7 Tests

Pure logic — generator's `IsGenerable`, recognizer's prefill decisions on synthetic `ElementTopology` snapshots — is unit-testable headlessly. Follow `Ponchevit.Tests/Revit/Families/MuroGeneratorSupportsTests.cs`.

The `Generate` and `Recognize` paths that actually touch RevitAPI are verified by **manual smoke test in Revit 2026**, which is the only integration verification path. There's no headless harness for RevitAPI.

### 7.8 Smoke-test checklist

For a new generator + recognizer, plan to verify all of:

1. **Build + deploy** succeeds (close Revit first if PostBuild fails on a locked DLL).
2. **Tab + ribbon** still appear; existing commands still work.
3. **Agregar Familia**:
   - Left-tree sections for the new element's subcapítulo are *enabled* (clickable, not greyed).
   - Cascade dropdowns drive a 10-digit code as expected; range TextBox shows up when `NumMin ≠ NumMax`.
   - Right panel shrinks correctly; constructible partidas are non-greyed; clicking "Seleccionar Partida" backfills the cascade.
   - Clicking "Agregar" creates the family type in the project; the 4 COVENIN shared params are populated on the new type; status line goes green.
   - Re-clicking with the same code re-uses the existing type rather than duplicating.
4. **Asignar Código** with a pre-selected element of the new category:
   - "Reconocer" populates the columns you decided are auto-detectable; qualitative ones stay greyed.
   - Manual selection in still-empty columns updates the live code display.
   - Clicking "Asignar" writes the 4 shared params on the element's *type* (not the instance); status line goes green.
5. **Codificación Dashboard**:
   - Placed instances of the new element are grouped by family type and appear in the list.
   - Codified rows render with their 10-digit code; non-codified rows say "Sin código".
   - Row's "Asignar código" button opens the Assign window pre-filled with that element.
   - "Generar Schedule" produces a `ViewSchedule` named `COVENIN - Codificación <timestamp>` that includes the new types.
6. **Save → close → reopen** the .rvt: COVENIN params, material mapping, schedules all survive intact.

If a step fails, the fix likely lives in the layer you most recently touched. Trace via `%AppData%\Ponchevit\log.txt` first.

### 7.9 ADRs for non-obvious choices

Any non-obvious decision you make while adding the module — "base RFA strategy for this element," "we chose to include qualitative column X as `Ambiguous` not `Undetectable` because…," "we expanded `ElementTopologyReader.ReadDimensions` to include `Ancho` for door types" — gets a same-day entry in `decisions.md`. Title with `### YYYY-MM-DD — Title`, body with `Decision / Rationale / Alternatives considered / Status`. The bar is "would someone six months from now wonder why this is the way it is?" If yes, log it.

---

## 8. Modifying an existing module

### 8.1 If the change is "behavior only" inside a generator/recognizer

Update the class, run the unit tests (headless), smoke-test in Revit, log a deviation if you changed something the docs assumed. No layer wiring is needed.

### 8.2 If the change touches the cascade UI

Most cascade-UI changes live in three files:

- `Ui/PartidaSelection/PartidaSelectionViewModel.cs` — the orchestration logic (filter, seed, append, backfill, build input).
- `Ui/PartidaSelection/CascadeRowViewModel.cs` — per-row state (selected option, range input, prefill state, material-link visibility).
- `Ui/PartidaSelection/PartidaSelectionWindow.xaml` — the three-panel layout and DataTriggers.

When you change cascade behavior, walk through *both* Generate mode and Assign mode by hand — the same VM serves both with a `Mode` flag, so a change in one can leak into the other. The mode behavioral matrix in roadmap §Phase 5 is the contract; respect it.

If a change requires touching `SetSelectedSilently`, `BackfillCascadeFromPath`, or `SeedRowsFromPath` — re-read ADR 2026-06-06 — Cascade seeding uses GetPath and ADR 2026-06-06 — ConfirmPartidaCommand replaces DataGrid-click backfill before editing. These three methods exist as they are because previous attempts at "simpler" approaches broke.

### 8.3 If the change touches Revit data

Walk the layers:

- New shared parameter? Add a GUID constant in `CoveninParameters.cs` (and update `SharedParameterWriter`). **Do not reuse a previously-defined GUID — issue a fresh one.** Document the addition in `revit-integration.md`. See ADR 2026-05-31 — GUID source-of-truth.
- New ExtensibleStorage payload? Evolve the existing schema via versioned JSON (Path B from ADR 2026-05-31 — Material mapping). Do **not** create a new Schema GUID.
- New schedule shape? Edit `CodificacionScheduleBuilder.Build`; remember it's fire-and-forget — the plugin doesn't track or update existing schedules. See ADR 2026-05-31 — Fire-and-forget schedules.

### 8.4 If the change touches the DAG or catalog

The DAG (`covenin.db`) and the catalog (`partidas.db`) are produced by the Python pipelines in `Partidas/` and `Tablas/`. They are **upstream**. Don't hand-edit the .db files in `Resources/`. If a code is wrong, the fix is in the pipeline; if the schema needs to change, bump `_meta.schema_version` and `ConnectionFactory.ExpectedSchemaVersion` together. See `data-layer.md` §2.

The plugin tolerates source-data anomalies (non-10-digit codes, `xxx` placeholder substrings) by logging and excluding them. It will not refuse to start. See ADR 2026-05-26 — Schema anomaly handling.

---

## 9. Tips, pitfalls, and shortcuts

### Reading the code

- Start at `App.cs` → `Composition/Services.cs` → the four Commands. Once those click, the dependency graph is no longer a maze.
- When a class name doesn't tell you what it does, the XML doc-comment at the top of the file usually does. If it doesn't, that's a bug — add one.
- ADRs (`decisions.md`) are *load-bearing*. Each documents why the obvious-looking alternative was rejected. When something seems weird, grep `decisions.md` before "fixing" it.

### Working in Revit

- **Always close Revit before rebuilding.** PostBuild fails with MSB3021 (locked DLL) otherwise. Not a code error.
- The COVENIN params are `TypeBinding`. They show up under "Edit Type… → Identity Data" in the Properties palette, *not* on the instance. Smoke-test `Type Properties` to confirm a write took.
- Schedules are user-owned. The plugin creates `COVENIN - Codificación <timestamp>` each click and forgets it. Rename, restyle, duplicate, delete — the plugin doesn't care.

### MVVM gotchas

- Modeless windows can't call `PickObject`. If you need element selection, the user pre-selects in Revit, or the Dashboard hands you the element. See ADR 2026-06-06 — No in-window element picker.
- `PostExternalEvent` is **queued, not synchronous**. Don't read the result of the delegate immediately after calling — use `Application.Current.Dispatcher.Invoke(…)` from inside the delegate to push status updates back to the VM. See `ui-patterns.md` §"Status message flow" for the pattern.
- During cascade backfill, use `SetSelectedSilently` to suppress per-row `PropertyChanged`. Otherwise each selection triggers the cascade handler mid-loop and corrupts state. See ADR 2026-06-06.
- The VM does not own a `Window` reference. To close, the VM raises `CloseRequested`; the window subscribes once in its constructor.

### Layer violations

- If `Domain.Tests` or `Data.Tests` won't compile because of a `using Autodesk.Revit.DB;` you accidentally added, congratulations — the test project just caught a layer violation. Move the offending Revit-touching code into `Revit/` and expose it through an interface.
- `Ui/` is checked by hand. If a VM file has `using Autodesk.Revit.DB;`, that's a bug. The right fix is almost always a long-accepting overload on the corresponding `Revit/` orchestrator (see `AssignCodeOrchestrator.Assign(Document, long, AssignInput)` for the pattern).

### Debugging

- `%AppData%\Ponchevit\log.txt` — single best diagnostic, always check first.
- `Composition/Services.cs` is the dependency graph; if a service shows up as `null`, you forgot to register it.
- Detach + re-attach the debugger when Revit reloads the addin between sessions.

### Performance

- The DAG is large (~377k connections) but lazy by `Parent_Id` with an in-memory cache. If the cascade panel feels slow, the cache isn't being hit — look at `SqliteCoveninRulesRepository.GetConexionesByParent`.
- `PartidaConstructibilityResolver` does a one-time DFS at startup with prefix-pruning. It completes in well under a second; if you change it and startup gets noticeably slow, you broke the pruning. See ADR 2026-06-06 — Constructibility eager-vs-lazy.

---

## 10. Examples — copy-paste templates

### 10.1 A new ribbon command

```csharp
// Commands/MyFeatureCommand.cs
using System;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Ponchevit.Ui.MyFeature;

namespace Ponchevit.Commands;

[Transaction(TransactionMode.Manual)]   // ReadOnly if you don't write
public class MyFeatureCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);

            var vm = new MyFeatureViewModel(/* services */);
            var window = new MyFeatureWindow(vm);
            window.Show();

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("MyFeature failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }
}
```

Register the button in `App.OnStartup`:

```csharp
PushButtonData btn = new PushButtonData(
    "MyFeature",                          // unique internal name — never change after release
    "My\nFeature",                        // ribbon label (\n splits to two lines)
    assemblyPath,
    "Ponchevit.Commands.MyFeatureCommand")
{
    ToolTip = "User-facing description."
};
panel.AddItem(btn);
```

### 10.2 A new pure-domain service

```csharp
// Domain/Foo/IFooService.cs
namespace Ponchevit.Domain.Foo;

public interface IFooService
{
    int DoWork(string input);
}

// Domain/Foo/FooService.cs
namespace Ponchevit.Domain.Foo;

public sealed class FooService : IFooService
{
    public int DoWork(string input)
    {
        // pure C# — no Revit, no Sqlite, no IO
        return input?.Length ?? 0;
    }
}
```

Register in `Composition/Services.cs`:

```csharp
// in fields
public IFooService FooService { get; }

// in constructor
FooService = fooService;

// in Build()
var foo = new FooService();
// pass to ctor: foo
```

### 10.3 A VM-to-Revit write under `PostExternalEvent`

```csharp
// inside a ViewModel command
[RelayCommand]
private void DoTheRevitThing()
{
    _revitContext.PostExternalEvent(doc =>
    {
        try
        {
            _orchestrator.Run(doc, /* args */);
            System.Windows.Application.Current?.Dispatcher.Invoke(() =>
            {
                StatusIsError = false;
                StatusMessage = "Listo.";
            });
        }
        catch (Exception ex)
        {
            _log.Error("DoTheRevitThing failed.", ex);
            var msg = ex.Message;
            System.Windows.Application.Current?.Dispatcher.Invoke(() =>
            {
                StatusIsError = true;
                StatusMessage = $"Error: {msg}";
            });
        }
    });
}
```

`_orchestrator` lives in `Revit/`, owns the transaction, accepts pure-C# arguments. The VM never names `Transaction`.

### 10.4 A new in-memory SQLite test fixture

```csharp
using Microsoft.Data.Sqlite;
using Xunit;
using Ponchevit.Data.Sqlite;

public class MyRepoTests
{
    private static SqliteConnection NewConn()
    {
        var c = new SqliteConnection("Data Source=:memory:");
        c.Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE Capitulos (id TEXT PRIMARY KEY, codigo TEXT, titulo TEXT);
            INSERT INTO Capitulos VALUES ('cap1', 'E4', 'Obras Arquitectónicas');
        ";
        cmd.ExecuteNonQuery();
        return c;
    }

    [Fact]
    public void Repo_ReadsCapitulos()
    {
        using var c = NewConn();
        var repo = /* construct repo passing c via a factory */;
        var caps = repo.GetCapitulos().ToList();
        Assert.Single(caps);
        Assert.Equal("E4", caps[0].Codigo);
    }
}
```

See `Ponchevit.Tests/Data/RepositoryTests.cs` for the production pattern (includes `_meta` schema-version setup).

---

## 11. Where to go next

Once you've finished this guide and you want depth:

1. Read `architecture.md` — same material, more detail on layers and the data plane.
2. Read `roadmap.md` — see what's done and what (if anything) is next.
3. Open `decisions.md` and skim every ADR title — each two-line title is itself a useful one-liner about how the project thinks. Read the body of any ADR whose title you don't already understand.
4. Smoke-test the four commands in Revit 2026 on a small `.rvt`. The flow makes much more sense after you've placed a wall and codified it.
5. If you're going to extend the generator catalog, re-read Section 7 with `MuroGenerator.cs` and `MuroRecognizer.cs` open side-by-side. They are the canonical references — every new module should look like them.

When the docs and the code disagree, the docs describe the intended target — flag the gap rather than silently re-deriving from code. That's the bargain that keeps this codebase legible to the next person.
