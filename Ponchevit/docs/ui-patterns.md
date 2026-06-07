# UI Patterns Reference

This document describes the patterns used in Ponchevit's WPF UI layer. A developer who reads this should be able to add a new ribbon command without re-reading Phase 3 or Phase 4 code.

---

## Modeless window lifecycle

Ponchevit windows are **modeless** — they use `Window.Show()`, not `Window.ShowDialog()`. This allows the user to continue working in Revit while the window is open.

### Show pattern

```csharp
// In the IExternalCommand.Execute method:
var vm = new MyViewModel(...);
var window = new MyWindow(vm);
window.Show();          // Returns immediately; Revit stays responsive.
return Result.Succeeded;
```

### VM-driven close

The VM exposes `CloseRequested` — an `EventHandler?`. The window subscribes in its constructor:

```csharp
public MyWindow(MyViewModel vm)
{
    InitializeComponent();
    DataContext = vm;
    vm.CloseRequested += (_, _) => Close();   // Window closes when VM fires the event.
}
```

The VM raises it from its Cancel (and optionally Save) commands:

```csharp
[RelayCommand]
private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);
```

Never call `Close()` directly from the VM — the VM must not hold a reference to the Window.

---

## CommunityToolkit.Mvvm conventions

All ViewModels inherit `ObservableObject`. Properties use source-generator attributes:

```csharp
public partial class MyViewModel : ObservableObject
{
    // ObservableProperty: generates the property, getter/setter, and OnXxxChanged partial.
    [ObservableProperty]
    private string _myField = string.Empty;

    // NotifyPropertyChangedFor: fires PropertyChanged for a computed property when the field changes.
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ComputedProp))]
    private bool _sourceField;

    public bool ComputedProp => _sourceField && SomeOtherCondition;

    // RelayCommand: generates a command from a private method.
    [RelayCommand]
    private void DoSomething() { ... }

    // CanExecute via a separate bool property:
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DoSomethingCommand))]
    private bool _canDoSomething;

    [RelayCommand(CanExecute = nameof(CanDoSomething))]
    private void DoSomething() { ... }
}
```

**Do not** hand-roll `INotifyPropertyChanged` or `ICommand` — always use the source generators.

---

## IRevitContext.PostExternalEvent — VM to RevitAPI bridging

WPF windows cannot call RevitAPI directly (they run on the UI thread, not Revit's main thread). The bridge is `IRevitContext.PostExternalEvent`:

```csharp
_revitContext.PostExternalEvent(doc =>
{
    // doc is a valid Document on Revit's main thread.
    using var t = new Transaction(doc, "My operation");
    t.Start();
    // ... RevitAPI calls ...
    t.Commit();
});
```

### When to use it

Any write (or API-guarded read) from a modeless window must go through `PostExternalEvent`. The delegate is queued onto Revit's external-event queue and executed at the next safe opportunity.

### What NOT to do

```csharp
// WRONG — PostExternalEvent queues the work; the window is still open.
// Do not wait for the result synchronously (it won't run until PostExternalEvent returns).
_revitContext.PostExternalEvent(doc => { myResult = ...; });
UseResult(myResult); // myResult is still default — race condition.
```

### Lifecycle

1. `RevitContextImpl` is created in `Services.Build()` — registers the `IExternalEventHandler` at startup.
2. Each `IExternalCommand.Execute` calls `services.RevitContext.Attach(commandData.Application)` to bind the current `UIApplication`.
3. Commands that open modeless windows do NOT own a transaction — the PostExternalEvent handler owns its own transaction.

---

## Two-stage material control

When a COVENIN cascade column is a **material** column (detected by `column.Nombre.Contains("MATERIAL")`), the central panel shows two stacked dropdowns instead of one:

1. **Stage 1 — Covenin material value**: the standard `ComboBox` from `CascadeMenuBuilder.Options`.
2. **Stage 2 — Revit material**: a second `ComboBox` showing only Revit project materials that map to the chosen Covenin value (sourced by inverting `IMaterialMappingRepository.GetAll()`).

When no Revit material maps to the selected Covenin value, a "Mapear material..." hyperlink is shown inline. Clicking it opens the `MaterialMappingWindow` **modeless** (on top of `AgregarFamiliaWindow` — both remain interactive). When the mapping window closes, the outer VM's `OnMaterialMappingClosed()` is called to refresh Stage 2.

The `CascadeRowViewModel.ShowMapearMaterialLink` flag controls this inline link's visibility.

### Adding a new material column (future capítulos)

No code changes are needed if the column name contains "MATERIAL" (case-insensitive). The two-stage control appears automatically.

---

## Left-tree → cascade seeding flow

Selecting any enabled node in the left tree (Capítulo, Subcapítulo, or Sección) seeds the cascade panel automatically:

1. **`TreeViewItem_Selected`** in the code-behind calls `SelectTreeNodeCommand`.
2. The VM calls **`SeedCascadeFromTreeNode(node)`**, which:
   a. Finds any constructible partida in the filtered list via `_constructibilityResolver.IsConstructible`.
   b. Retrieves the pre-computed root-to-leaf connection-ID path via `_constructibilityResolver.GetPath(seedPartida.CodigoPartida)`.
   c. Calls **`SeedRowsFromPath(nodeCode, fullPath)`**, which walks the path step by step, adding each row as `IsSeeded = true` and stopping when the assembled code equals the node's code.
   d. After the seeded prefix is exhausted, `AppendNextCascadeLevel` adds the first user-editable row.
3. Seeded rows appear with a light-blue background and an "(auto)" badge. They are **locked** (`IsEnabled = false`) — the user changes the tree node, not the seeded rows.
4. The **`CurrentCodeDisplay`** property updates to e.g. `E411XXXXXX`.

**Why path-based, not greedy?** Empty-bridge connections (`CodigoAportado = ""`) are structurally valid DAG edges that contribute nothing to the assembled code. A greedy prefix-match filter (checking `nodeCode.StartsWith(assembled + option.CodigoAportado)`) matches every option when `CodigoAportado` is empty, breaking disambiguation. `GetPath` is pre-computed by the DFS in `PartidaConstructibilityResolver` and is always correct. See ADR 2026-06-06 for full rationale.

### What if the tree node has no DAG rules?

For nodes outside Capítulo E4, no constructible partida is found, and `HasNoRules` is set to true showing "Sin reglas COVENIN para este capítulo". The right panel still shows the filtered partidas (all greyed-out because they are unconstructible outside E4).

---

## Code-being-built display

The central panel always shows a live code display at the top:

```
Código: E411XXXXXX
```

- Rendered in monospace bold via `{Binding CurrentCodeDisplay}`.
- Updated by `UpdateCodeDisplay()` after every cascade selection change.
- `CurrentCodeDisplay = AssembleCodeFromPath().PadRight(10, 'X')`.
- An empty path produces `XXXXXXXXXX`.
- A fully-resolved path produces the exact 10-digit code.

---

## Click-partida backfill flow

The right-panel DataGrid uses a **two-step** selection model to avoid accidental backfill:

1. **Row click** (`PartidasDataGrid_SelectionChanged` in code-behind) sets `_vm.SelectedPartidaItem = item` only. No cascade change.
2. **"Seleccionar Partida" button** (`ConfirmPartidaCommand`) performs the actual backfill. It is only enabled when `SelectedPartidaItem?.IsConstructible == true`.

When `ConfirmPartidaCommand` executes:
1. The VM calls `_constructibilityResolver.GetPath(SelectedPartidaItem.CodigoPartida)` to retrieve the ordered connection-ID list.
2. **`BackfillCascadeFromPath(path)`** rebuilds `CascadeRows` from scratch: for each connection ID in the path, the correct `MenuLevel` is fetched, a non-seeded `CascadeRowViewModel` is constructed, and `SetSelectedSilently(option)` is called to set the selection without triggering cascade rebuilds.
3. After backfill: `UpdateCodeDisplay()` → `CurrentCodeDisplay` shows the full 10-digit code; `UpdateCanAgregar()` → `CanAgregar` becomes true; the right panel filter narrows to just the selected partida.

### Why SetSelectedSilently?

During backfill all rows are built and selected in a single loop. If each `SelectedOption = x` raised PropertyChanged normally, the VM's cascade handler would truncate and rebuild the rows below — corrupting state mid-loop. `SetSelectedSilently` suppresses PropertyChanged during the set and fires it manually afterward so WPF bindings update without triggering the cascade handler.

---

## Greyed-partida semantics

Every item in the right panel is a `PartidaDisplayItem` (not a raw `Partida`). It carries:

- `Partida` — the underlying domain object.
- `IsConstructible` — whether a DAG path producing this code exists.

In the XAML:

```xml
<DataGrid.RowStyle>
    <Style TargetType="DataGridRow">
        <Style.Triggers>
            <DataTrigger Binding="{Binding IsConstructible}" Value="False">
                <Setter Property="Foreground" Value="LightGray"/>
                <Setter Property="IsEnabled" Value="False"/>
                <Setter Property="ToolTip" Value="Esta partida no puede construirse…"/>
            </DataTrigger>
        </Style.Triggers>
    </Style>
</DataGrid.RowStyle>
```

- Unconstructible rows are visually greyed-out and disabled (no selection, no backfill).
- They remain visible so the user knows they exist in the flat catalog.
- Outside Capítulo E4, all partidas are unconstructible (no DAG rules).
- Roughly 1/3 of E4 partidas are unconstructible due to typos in the source list.

**Computing constructibility:** `PartidaConstructibilityResolver` in `Domain/Catalog/` pre-computes this at `Services.Build()` time via a DFS over the DAG with prefix-pruning. The result is a `HashSet<string>` for O(1) lookup. See ADR 2026-06-06 for details.

---

## Registry-driven left tree enabling/disabling

The left tree (Capítulo → Subcapítulo → Sección) enables only the Sección nodes whose category has a registered `IFamilyGenerator`.

### How it works

1. `AgregarFamiliaViewModel.BuildTree()` iterates all `IFamilyGenerator[]` from `Services.FamilyGenerators` and collects their `SupportedCategory` values.
2. For each `Seccion`, `IsSeccionEnabled(sec, supportedCategories)` returns `true` only if the sección's code prefix maps to a supported category. For MVP: prefix `"E41"` → `OST_Walls`.
3. `TreeNodeViewModel.IsEnabled = false` causes the UI to render that node greyed-out (via a `DataTrigger` on the `IsEnabled` binding).
4. Clicking a disabled node is a no-op — `SelectTreeNodeCommand` checks `node.IsEnabled` before proceeding.

### Adding a new generator

1. Create `Revit/Families/XxxGenerator.cs` implementing `IFamilyGenerator`.
2. Register in `Services.Build()`: add to the `generators` array.
3. Add a case in `IsSeccionEnabled` mapping the new sección prefix to the new category.

---

## Numeric range inputs

Dimensional columns in the COVENIN DAG (e.g. espesor, altura, longitud) are always rendered as **ComboBoxes** showing the DAG option labels (normative text). When the selected option has `NumMin ≠ NumMax` — a true range, e.g. "5–30 cm" — an additional `TextBox` appears below the ComboBox for the user to enter a concrete value.

In `CascadeRowViewModel`:
- `SelectedOptionIsRange` — computed: `NumMin.HasValue && NumMax.HasValue && NumMin != NumMax`.
- `RangeInput` — `double?` observable; the TextBox is bound to this with `UpdateSourceTrigger=LostFocus`.
- `RangeMin`, `RangeMax` — displayed as hint text below the TextBox.
- `RangeUnit` — unit label from `SelectedOption.ValorData.Unidad` (e.g. "cm").

In the XAML, the range TextBox StackPanel is bound to `SelectedOptionIsRange` with a `BoolToVisibilityConverter`. It appears for both material and non-material columns.

`CanAgregar` requires `!r.SelectedOptionIsRange || r.RangeInput.HasValue` for every cascade row — a range option without a typed value blocks the Agregar button.

`BuildGeneratorInput()` populates `NumericValues` (IdColumna → metres) only for range rows with a typed value. For exact-value options (`NumMin == NumMax`), the value is already in `SelectedValores.ValorData.NumMin` — `MuroGenerator` reads it from there as a fallback.

### Exact-value options (no TextBox)

When `NumMin == NumMax` (e.g. espesor "15 cm"), the ComboBox is the only control. The user selects it like any non-numeric option. `MuroGenerator` reads `valor.NumMin` directly from the selected `Valor` object — no user input needed.

---

## Status message flow

After the user clicks "Agregar", a status label at the bottom-left of the window shows feedback:

- **Success** (green): "Familia creada correctamente."
- **Error** (red, semi-bold): "Error: {exception message}"

The flow:

1. VM's `Agregar()` clears `StatusMessage` and posts the external event.
2. Inside the PostExternalEvent delegate (Revit thread), the orchestrator runs.
3. On success: `Application.Current.Dispatcher.Invoke(() => { StatusIsError = false; StatusMessage = "..."; })`.
4. On failure: same dispatch with `StatusIsError = true` and the exception message.
5. XAML `DataTrigger` on `StatusIsError` switches the `TextBlock.Foreground` between green and red.

The VM owns `StatusMessage` and `StatusIsError` as `[ObservableProperty]` booleans/strings. The window code-behind has no involvement in status routing.

---

## Project-material query service

Both `AgregarFamiliaCommand` and `MapeoMaterialesCommand` use `IProjectMaterialQuery` (implemented by `ProjectMaterialQuery` in `Revit/`) to collect material names from the active document.

```csharp
// In the Command layer:
var materials = services.ProjectMaterialQuery.GetProjectMaterials(doc);
// Pass to VM constructor or MaterialMappingViewModel constructor.
```

The window code-behind may also call `IProjectMaterialQuery` when refreshing materials after the mapping window closes — it receives the service via constructor injection. The window never references `FilteredElementCollector` or `Material` directly.

### Pattern rule

Window code-behind: routes events between View and VM only. No RevitAPI calls. If project materials are needed in the window, accept `IProjectMaterialQuery` in the constructor and call it.

---

## Ribbon command registration

All commands follow this pattern in `App.OnStartup`:

```csharp
PushButtonData btn = new PushButtonData(
    "UniqueName",        // internal name — never changes after registration
    "Display\nText",     // shown on ribbon button (newline is used for two-line labels)
    assemblyPath,
    "Ponchevit.Commands.MyCommand");
btn.ToolTip = "User-facing description.";
panel.AddItem(btn);
```

Commands are registered in `Commands/MyCommand.cs` and must be decorated with `[Transaction(TransactionMode.Manual)]` (or `ReadOnly` if no writes). They call `services.RevitContext.Attach(commandData.Application)` before creating windows.
