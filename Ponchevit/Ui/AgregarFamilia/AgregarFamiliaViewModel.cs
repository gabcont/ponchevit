using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Ponchevit.Data;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Materials;
using Ponchevit.Domain.Model;
using Ponchevit.Domain.Query;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.Context;
using Ponchevit.Revit.Families;
using Ponchevit.Ui.Common;
using Ponchevit.Ui.MaterialMapping;

namespace Ponchevit.Ui.AgregarFamilia;

/// <summary>
/// ViewModel for the Agregar Familia three-panel window.
///
/// Left tree:    Capítulo → Subcapítulo → Sección (only those with a generator are enabled).
///               Selecting a node seeds the cascade panel with the matching prefix values
///               and updates <see cref="CurrentCodeDisplay"/>.
///
/// Central panel: cascading COVENIN DAG dropdowns, with two-stage material control and
///               numeric TextBox for dimensional columns.
///               Each selection narrows the right-panel partida list and updates the
///               live code display (e.g. "E411XXXXXX"). Seeded dropdowns from a tree
///               selection are locked to prevent accidental override.
///
/// Right panel:  <see cref="PartidaDisplayItems"/> — always populated; shrinks as tree
///               and cascade selections narrow the code prefix. Unconstructible partidas
///               appear greyed-out. Clicking "Seleccionar Partida" on a constructible
///               partida backfills the cascade dropdowns with the DAG path that produces
///               its code.
///
/// Fix #1: no RevitAPI types in this file. Transactions are owned by
///         <see cref="FamilyGenerationOrchestrator"/>, which is called via PostExternalEvent.
/// Fix #2: cascade path tracks (IdConexion, CodigoAportado) pairs — no repo re-query.
/// Fix #3: shared-param hierarchy titles derived via PartidaHierarchyResolver from the
///         assembled 10-digit code; no substring-hacking fallbacks.
/// Fix #9: StatusMessage observable property surfaced to the UI for error / success feedback.
/// Issue 2g: Largo/Altura removed; Descripcion derived from catalog; SelectedValores and
///           NumericValues built from CascadeRow state; ConfirmPartidaCommand replaces
///           SelectPartidaCommand backfill trigger.
/// Issue 3: SelectedPartidaItem setter is plain; ConfirmPartidaCommand performs backfill.
/// </summary>
public sealed partial class AgregarFamiliaViewModel : ObservableObject
{
    // ── Injected services ────────────────────────────────────────────────────
    private readonly IPartidasRepository _partidasRepo;
    private readonly ICoveninRulesRepository _rulesRepo;
    private readonly PartidaCatalog _catalog;
    private readonly PartidaConstructibilityResolver _constructibilityResolver;
    private readonly IMaterialMappingResolver _materialResolver;
    private readonly IMaterialMappingRepository _materialMappingRepo;
    private readonly IRevitContext _revitContext;
    private readonly IFamilyGenerator[] _generators;
    private readonly FamilyGenerationOrchestrator _orchestrator;
    private readonly PartidaHierarchyResolver _hierarchyResolver;
    private readonly ILog _log;

    // Project materials list (passed from the Command — mirrors Phase 3 pattern).
    private readonly IReadOnlyList<string> _projectMaterials;

    // ── State ────────────────────────────────────────────────────────────────
    private readonly CascadeMenuBuilder _cascadeBuilder;
    private readonly FilteredPartidaCollection _filteredPartidas;

    // Fix #2: track (IdConexion, CodigoAportado) pairs instead of just IDs.
    // CodigoAportado is cached from the MenuOption at selection time so
    // AssembleCodeFromPath never needs to re-query the repository.
    private readonly List<(string IdConexion, string CodigoAportado)> _selectedConnectionPath = new();

    // Number of cascade rows seeded from the left-tree selection (locked in UI).
    private int _seededRowCount;

    // ── Observable UI state ──────────────────────────────────────────────────
    public ObservableCollection<TreeNodeViewModel> TreeRoots { get; } = new();
    public ObservableCollection<CascadeRowViewModel> CascadeRows { get; } = new();

    /// <summary>
    /// Filtered + constructibility-annotated partidas for the right panel.
    /// Rebuilt whenever the underlying filter or cascade selection changes.
    /// </summary>
    public ObservableCollection<PartidaDisplayItem> PartidaDisplayItems { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanAgregar))]
    private TreeNodeViewModel? _selectedTreeNode;

    [ObservableProperty]
    private string _noRulesMessage = string.Empty;

    [ObservableProperty]
    private bool _hasNoRules;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(AgregarCommand))]
    private bool _canAgregar;

    /// <summary>
    /// Live code display shown at the top of the central panel, e.g. "E411XXXXXX".
    /// Filled positions replace X as cascade dropdowns are selected.
    /// </summary>
    [ObservableProperty]
    private string _currentCodeDisplay = new string('X', 10);

    /// <summary>
    /// Currently selected item in the right-panel DataGrid.
    /// Plain setter — no backfill side-effect. Use ConfirmPartidaCommand to backfill.
    /// </summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ConfirmPartidaCommand))]
    private PartidaDisplayItem? _selectedPartidaItem;

    // Fix #9: user-facing status feedback after Agregar.
    [ObservableProperty]
    private string? _statusMessage;

    [ObservableProperty]
    private bool _statusIsError;

    public event EventHandler? CloseRequested;
    /// <summary>Raised when the "Mapear Material" link is clicked; subscribers open the mapping window.</summary>
    public event EventHandler? MapearMaterialRequested;

    // ── Constructor ──────────────────────────────────────────────────────────
    public AgregarFamiliaViewModel(
        IPartidasRepository partidasRepo,
        ICoveninRulesRepository rulesRepo,
        PartidaCatalog catalog,
        PartidaConstructibilityResolver constructibilityResolver,
        IMaterialMappingResolver materialResolver,
        IMaterialMappingRepository materialMappingRepo,
        IRevitContext revitContext,
        IFamilyGenerator[] generators,
        FamilyGenerationOrchestrator orchestrator,
        IReadOnlyList<string> projectMaterials,
        PartidaHierarchyResolver hierarchyResolver,
        ILog log)
    {
        _partidasRepo             = partidasRepo             ?? throw new ArgumentNullException(nameof(partidasRepo));
        _rulesRepo                = rulesRepo                ?? throw new ArgumentNullException(nameof(rulesRepo));
        _catalog                  = catalog                  ?? throw new ArgumentNullException(nameof(catalog));
        _constructibilityResolver = constructibilityResolver ?? throw new ArgumentNullException(nameof(constructibilityResolver));
        _materialResolver         = materialResolver         ?? throw new ArgumentNullException(nameof(materialResolver));
        _materialMappingRepo      = materialMappingRepo      ?? throw new ArgumentNullException(nameof(materialMappingRepo));
        _revitContext             = revitContext             ?? throw new ArgumentNullException(nameof(revitContext));
        _generators               = generators               ?? Array.Empty<IFamilyGenerator>();
        _orchestrator             = orchestrator             ?? throw new ArgumentNullException(nameof(orchestrator));
        _projectMaterials         = projectMaterials         ?? Array.Empty<string>();
        _hierarchyResolver        = hierarchyResolver        ?? throw new ArgumentNullException(nameof(hierarchyResolver));
        _log                      = log                      ?? throw new ArgumentNullException(nameof(log));

        _cascadeBuilder   = new CascadeMenuBuilder(rulesRepo);
        _filteredPartidas = new FilteredPartidaCollection(catalog.GetPartidas());

        BuildTree();
        RebuildDisplayItems();
    }

    // ── Tree building ────────────────────────────────────────────────────────

    private void BuildTree()
    {
        var capitulos    = _partidasRepo.GetCapitulos().OrderBy(c => c.Codigo).ToList();
        var subcapitulos = _partidasRepo.GetSubcapitulos().ToList();
        var secciones    = _partidasRepo.GetSecciones().ToList();

        foreach (var cap in capitulos)
        {
            var capNode = new TreeNodeViewModel(
                cap.Titulo, cap.Codigo, TreeNodeKind.Capitulo, isEnabled: true);

            var subs = subcapitulos.Where(s => s.CapituloId == cap.Id)
                                   .OrderBy(s => s.Codigo)
                                   .ToList();
            foreach (var sub in subs)
            {
                var subNode = new TreeNodeViewModel(
                    sub.Titulo, sub.Codigo, TreeNodeKind.Subcapitulo, isEnabled: true);

                var secs = secciones.Where(s => s.SubcapituloId == sub.Id)
                                    .OrderBy(s => s.Codigo)
                                    .ToList();
                foreach (var sec in secs)
                {
                    // A sección is enabled when at least one registered generator supports its code.
                    // This keeps BuiltInCategory entirely inside Revit/ — the VM never sees it.
                    bool enabled = _generators.Any(g => g.IsGenerable(sec.Codigo));
                    var secNode = new TreeNodeViewModel(
                        sec.Titulo, sec.Codigo, TreeNodeKind.Seccion, isEnabled: enabled);
                    subNode.Children.Add(secNode);
                }

                capNode.Children.Add(subNode);
            }

            TreeRoots.Add(capNode);
        }
    }

    // ── Tree selection ───────────────────────────────────────────────────────

    [RelayCommand]
    public void SelectTreeNode(TreeNodeViewModel? node)
    {
        if (node == null || !node.IsEnabled) return;

        SelectedTreeNode = node;

        string? capCodigo = null, subCodigo = null, secCodigo = null;
        switch (node.Kind)
        {
            case TreeNodeKind.Capitulo:    capCodigo = node.Codigo; break;
            case TreeNodeKind.Subcapitulo: subCodigo = node.Codigo; break;
            case TreeNodeKind.Seccion:     secCodigo = node.Codigo; break;
        }

        _filteredPartidas.ApplyFilter(capCodigo, subCodigo, secCodigo);

        _selectedConnectionPath.Clear();
        _seededRowCount = 0;
        SeedCascadeFromTreeNode(node);
        UpdateCodeDisplay();
        RebuildDisplayItems();
        UpdateCanAgregar();
    }

    // ── Cascade seeding from tree node ───────────────────────────────────────

    private void SeedCascadeFromTreeNode(TreeNodeViewModel node)
    {
        CascadeRows.Clear();
        _selectedConnectionPath.Clear();
        _seededRowCount = 0;

        string? nodeCode = node.Codigo;
        if (string.IsNullOrEmpty(nodeCode))
        {
            RebuildCascade(parentConnectionId: null);
            return;
        }

        // Find any constructible partida whose code starts with the node code.
        // _filteredPartidas is already filtered to the node's subtree.
        var seedPartida = _filteredPartidas.Items
            .FirstOrDefault(p => _constructibilityResolver.CanBeConstructed(p.CodigoPartida));

        if (seedPartida == null)
        {
            HasNoRules     = true;
            NoRulesMessage = "Sin reglas COVENIN para este capítulo";
            return;
        }

        var fullPath = _constructibilityResolver.GetPath(seedPartida.CodigoPartida);
        if (fullPath == null || fullPath.Count == 0)
        {
            HasNoRules     = true;
            NoRulesMessage = "Sin reglas COVENIN para este capítulo";
            return;
        }

        HasNoRules     = false;
        NoRulesMessage = string.Empty;

        SeedRowsFromPath(nodeCode, fullPath);
    }

    private void SeedRowsFromPath(string nodeCode, IReadOnlyList<string> fullPath)
    {
        string? currentParent = null;
        var assembled = new StringBuilder();

        foreach (string connId in fullPath)
        {
            var level = _cascadeBuilder.GetNextLevel(currentParent);
            if (level == null) break;

            var option = level.Options.FirstOrDefault(o => o.IdConexion == connId);
            if (option == null) break;

            bool isMaterial = level.Columna.Nombre.Contains("MATERIAL", StringComparison.OrdinalIgnoreCase);
            var rowVm = new CascadeRowViewModel(
                level.Columna.Nombre,
                level.Columna.IdColumna,
                level.Options,
                isMaterial,
                isSeeded: true,
                isNumericColumn: level.IsNumericColumn);

            rowVm.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(CascadeRowViewModel.SelectedOption))
                    OnCascadeSelectionChanged(rowVm);
            };
            if (isMaterial)
                rowVm.MapearMaterialRequested += (_, _) => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

            rowVm.SetSelectedSilently(option);
            string contrib = option.CodigoAportado ?? string.Empty;
            _selectedConnectionPath.Add((connId, contrib));
            _seededRowCount++;
            CascadeRows.Add(rowVm);

            if (contrib.Length > 0)
                assembled.Append(contrib);

            if (assembled.ToString().Equals(nodeCode, StringComparison.OrdinalIgnoreCase))
            {
                AppendNextCascadeLevel(connId);
                return;
            }

            currentParent = connId;
        }
    }

    private void AddNormalCascadeRow(MenuLevel level, string? parentConnectionId)
    {
        _ = parentConnectionId;
        bool isMaterial = level.Columna.Nombre.Contains("MATERIAL", StringComparison.OrdinalIgnoreCase);
        var rowVm = new CascadeRowViewModel(
            level.Columna.Nombre,
            level.Columna.IdColumna,
            level.Options,
            isMaterial,
            isSeeded: false,
            isNumericColumn: level.IsNumericColumn);

        rowVm.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(CascadeRowViewModel.SelectedOption))
                OnCascadeSelectionChanged(rowVm);
        };

        if (isMaterial)
            rowVm.MapearMaterialRequested += (_, _) => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

        CascadeRows.Add(rowVm);
    }

    // ── Cascade panel ────────────────────────────────────────────────────────

    private void RebuildCascade(string? parentConnectionId)
    {
        CascadeRows.Clear();

        if (SelectedTreeNode == null) return;

        var level = _cascadeBuilder.GetNextLevel(parentConnectionId);
        if (level == null)
        {
            bool noRulesForCapitulo = parentConnectionId == null;
            HasNoRules     = noRulesForCapitulo;
            NoRulesMessage = noRulesForCapitulo
                ? "Sin reglas COVENIN para este capítulo"
                : string.Empty;
            return;
        }

        HasNoRules     = false;
        NoRulesMessage = string.Empty;

        AddNormalCascadeRow(level, parentConnectionId);
    }

    private void OnCascadeSelectionChanged(CascadeRowViewModel changedRow)
    {
        if (changedRow.SelectedOption == null) return;

        int rowIndex = CascadeRows.IndexOf(changedRow);
        if (rowIndex < 0) return;

        // Fix #2: rebuild path from the rows above the changed row (all have SelectedOption set).
        _selectedConnectionPath.Clear();
        for (int i = 0; i < rowIndex; i++)
        {
            var prevRow = CascadeRows[i];
            if (prevRow.SelectedOption != null)
                _selectedConnectionPath.Add(
                    (prevRow.SelectedOption.IdConexion,
                     prevRow.SelectedOption.CodigoAportado ?? string.Empty));
        }
        // Add the newly selected row.
        _selectedConnectionPath.Add(
            (changedRow.SelectedOption.IdConexion,
             changedRow.SelectedOption.CodigoAportado ?? string.Empty));

        while (CascadeRows.Count > rowIndex + 1)
            CascadeRows.RemoveAt(CascadeRows.Count - 1);

        if (changedRow.IsMaterialColumn)
            RefreshRevitMaterials(changedRow);

        AppendNextCascadeLevel(changedRow.SelectedOption.IdConexion);

        UpdateCodeDisplay();
        UpdateRightPanelFromPath();
        RebuildDisplayItems();
        UpdateCanAgregar();
    }

    private void AppendNextCascadeLevel(string? parentConnectionId)
    {
        var level = _cascadeBuilder.GetNextLevel(parentConnectionId);
        if (level == null) return;

        bool isMaterial = level.Columna.Nombre.Contains("MATERIAL", StringComparison.OrdinalIgnoreCase);
        var rowVm = new CascadeRowViewModel(
            level.Columna.Nombre,
            level.Columna.IdColumna,
            level.Options,
            isMaterial,
            isSeeded: false,
            isNumericColumn: level.IsNumericColumn);

        rowVm.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(CascadeRowViewModel.SelectedOption))
                OnCascadeSelectionChanged(rowVm);
        };

        if (isMaterial)
            rowVm.MapearMaterialRequested += (_, _) => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

        CascadeRows.Add(rowVm);
    }

    private void RefreshRevitMaterials(CascadeRowViewModel row)
    {
        if (row.SelectedIdValor == null)
        {
            row.SetRevitMaterials(Enumerable.Empty<RevitMaterialOption>());
            return;
        }

        var allMappings = _materialMappingRepo.GetAll();
        var matching = allMappings
            .Where(kvp => string.Equals(kvp.Value, row.SelectedIdValor, StringComparison.OrdinalIgnoreCase))
            .Select(kvp => new RevitMaterialOption(kvp.Key, kvp.Key))
            .OrderBy(m => m.Display, StringComparer.OrdinalIgnoreCase)
            .ToList();

        row.SetRevitMaterials(matching);
    }

    private void UpdateRightPanelFromPath()
    {
        if (SelectedTreeNode == null) return;

        string? secCodigo = SelectedTreeNode.Kind == TreeNodeKind.Seccion    ? SelectedTreeNode.Codigo : null;
        string? subCodigo = SelectedTreeNode.Kind == TreeNodeKind.Subcapitulo ? SelectedTreeNode.Codigo : null;
        string? capCodigo = SelectedTreeNode.Kind == TreeNodeKind.Capitulo   ? SelectedTreeNode.Codigo : null;

        string? mask = BuildCodeMask();
        _filteredPartidas.ApplyFilter(capCodigo, subCodigo, secCodigo, codePrefix: mask);
    }

    // ── Code display ─────────────────────────────────────────────────────────

    private void UpdateCodeDisplay()
    {
        string assembled = AssembleCodeFromPath();
        int filled = Math.Min(assembled.Length, 10);
        CurrentCodeDisplay = assembled[..filled].PadRight(10, 'X');
    }

    /// <summary>
    /// Fix #2: assembles the code from cached CodigoAportado values in the path.
    /// No repository re-query.
    /// </summary>
    private string AssembleCodeFromPath()
    {
        var sb = new StringBuilder();
        foreach (var (_, codigoAportado) in _selectedConnectionPath)
        {
            if (codigoAportado.Length > 0 && sb.Length < 10)
            {
                int room = 10 - sb.Length;
                sb.Append(codigoAportado.Length <= room ? codigoAportado : codigoAportado[..room]);
            }
        }
        return sb.ToString();
    }

    private string? BuildCodeMask()
    {
        if (_selectedConnectionPath.Count == 0) return null;
        string assembled = AssembleCodeFromPath();
        if (assembled.Length == 0) return null;
        return assembled.PadRight(10, 'X');
    }

    // ── Right panel display items ─────────────────────────────────────────────

    private void RebuildDisplayItems()
    {
        PartidaDisplayItems.Clear();
        foreach (var p in _filteredPartidas.Items)
        {
            bool canBeConstructed = _constructibilityResolver.CanBeConstructed(p);
            PartidaDisplayItems.Add(new PartidaDisplayItem(p, canBeConstructed));
        }
    }

    // ── Partida confirmation — backfill cascade ───────────────────────────────

    /// <summary>
    /// Confirms the selected partida and backfills the cascade with its DAG path.
    /// Only enabled when a constructible partida is selected.
    /// (Issue 3: renamed from SelectPartidaCommand; selection-only now done by setting SelectedPartidaItem.)
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanConfirmPartida))]
    public void ConfirmPartida()
    {
        if (SelectedPartidaItem == null || !SelectedPartidaItem.CanBeConstructed) return;

        var path = _constructibilityResolver.GetPath(SelectedPartidaItem.CodigoPartida);
        if (path == null || path.Count == 0) return;

        BackfillCascadeFromPath(path);
        UpdateCodeDisplay();
        _filteredPartidas.ApplyFilter(codePrefix: SelectedPartidaItem.CodigoPartida);
        RebuildDisplayItems();
        UpdateCanAgregar();
    }

    private bool CanConfirmPartida()
        => SelectedPartidaItem?.CanBeConstructed == true;

    private void BackfillCascadeFromPath(IReadOnlyList<string> connectionIds)
    {
        CascadeRows.Clear();
        _selectedConnectionPath.Clear();
        _seededRowCount = 0;

        string? currentParent = null;
        foreach (var connId in connectionIds)
        {
            var level = _cascadeBuilder.GetNextLevel(currentParent);
            if (level == null) break;

            bool isMaterial = level.Columna.Nombre.Contains("MATERIAL", StringComparison.OrdinalIgnoreCase);
            var rowVm = new CascadeRowViewModel(
                level.Columna.Nombre,
                level.Columna.IdColumna,
                level.Options,
                isMaterial,
                isSeeded: false,
                isNumericColumn: level.IsNumericColumn);

            rowVm.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(CascadeRowViewModel.SelectedOption))
                    OnCascadeSelectionChanged(rowVm);
            };

            if (isMaterial)
                rowVm.MapearMaterialRequested += (_, _) => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

            var option = level.Options.FirstOrDefault(o => o.IdConexion == connId);
            if (option != null)
            {
                rowVm.SetSelectedSilently(option);
                // Fix #2: cache CodigoAportado from the MenuOption.
                _selectedConnectionPath.Add((connId, option.CodigoAportado ?? string.Empty));

                if (isMaterial)
                    RefreshRevitMaterials(rowVm);
            }

            CascadeRows.Add(rowVm);
            currentParent = connId;
        }

        HasNoRules     = false;
        NoRulesMessage = string.Empty;
    }

    // ── "Agregar" enablement ─────────────────────────────────────────────────

    private void UpdateCanAgregar()
    {
        if (_selectedConnectionPath.Count == 0)
        {
            CanAgregar = false;
            return;
        }

        bool allRowsSelected = CascadeRows.All(r =>
            r.SelectedOption != null
            && (!r.SelectedOptionIsRange || r.RangeInput.HasValue));
        if (!allRowsSelected)
        {
            CanAgregar = false;
            return;
        }

        string assembled = AssembleCodeFromPath();
        CanAgregar = assembled.Length == 10;
    }

    // ── "Agregar" command ────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanAgregar))]
    private void Agregar()
    {
        if (SelectedTreeNode == null && _selectedConnectionPath.Count == 0) return;

        var generator = ResolveGenerator();
        if (generator == null)
        {
            _log.Warn("No generator found for selected tree node.");
            return;
        }

        var input = BuildGeneratorInput();
        if (input == null) return;

        var capturedInput     = input;
        var capturedGenerator = generator;
        var capturedOrchestrator = _orchestrator;

        // Fix #9: clear any previous status.
        StatusMessage = null;
        StatusIsError = false;

        // Fix #1: the VM now delegates all RevitAPI work (including Transaction) to
        // FamilyGenerationOrchestrator — no Autodesk.Revit.DB types instantiated here.
        _revitContext.PostExternalEvent(doc =>
        {
            try
            {
                capturedOrchestrator.Generate(doc, capturedGenerator, capturedInput);

                // Fix #9: marshal success status back to the UI thread.
                Application.Current?.Dispatcher.Invoke(() =>
                {
                    StatusIsError = false;
                    StatusMessage = "Familia creada correctamente.";
                });
            }
            catch (Exception ex)
            {
                _log.Error("Agregar Familia: generation failed.", ex);

                // Fix #9: marshal error message back to the UI thread.
                var userMessage = ex.Message;
                Application.Current?.Dispatcher.Invoke(() =>
                {
                    StatusIsError = true;
                    StatusMessage = $"Error: {userMessage}";
                });
            }
        });
    }

    [RelayCommand]
    private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);

    [RelayCommand]
    private void OpenMapearMaterial(CascadeRowViewModel? row)
        => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

    // ── Helper: resolve generator ─────────────────────────────────────────

    private IFamilyGenerator? ResolveGenerator()
    {
        string? prefix = SelectedTreeNode?.Codigo;

        if (prefix == null && _selectedConnectionPath.Count > 0)
            prefix = AssembleCodeFromPath();

        if (prefix == null) return null;

        // Delegate category knowledge to the generator strategy — no BuiltInCategory in Ui/.
        return _generators.FirstOrDefault(g => g.IsGenerable(prefix));
    }

    // ── Helper: build GeneratorInput ─────────────────────────────────────

    /// <summary>
    /// Builds the GeneratorInput from current cascade state.
    ///
    /// Fix #3: derives shared-param values from the assembled 10-digit code via
    /// PartidaHierarchyResolver. The 3 hierarchy params store Spanish titles
    /// (human-readable in the Revit Properties panel). CodigoCompletoName stores
    /// the full 10-digit code. No substring-hacking fallbacks.
    ///
    /// Issue 2g: Descripcion is looked up from the catalog; SelectedValores and
    /// NumericValues are derived from CascadeRow state.
    /// </summary>
    private GeneratorInput? BuildGeneratorInput()
    {
        string assembledCode = AssembleCodeFromPath();
        if (string.IsNullOrEmpty(assembledCode)) return null;

        var codigo = new CodigoCovenin(assembledCode);

        var (capitulo, subcapitulo, seccion) = _hierarchyResolver.Resolve(assembledCode);

        string capituloTitle    = capitulo?.Titulo    ?? string.Empty;
        string subcapituloTitle = subcapitulo?.Titulo ?? string.Empty;
        string seccionTitle     = seccion?.Titulo     ?? string.Empty;

        // Look up Descripcion from the partida catalog.
        string descripcion = _filteredPartidas.Items
            .FirstOrDefault(p => string.Equals(p.CodigoPartida, assembledCode, StringComparison.OrdinalIgnoreCase))
            ?.Descripcion
            ?? assembledCode;

        var coveninValues = CascadeRows
            .Where(r => r.SelectedIdValor != null)
            .ToDictionary(r => r.IdColumna, r => r.SelectedIdValor!);

        // SelectedValores: IdColumna → Valor (full object) for rows that have ValorData.
        var selectedValores = CascadeRows
            .Where(r => r.SelectedOption?.ValorData != null)
            .ToDictionary(r => r.IdColumna, r => r.SelectedOption!.ValorData!);

        // NumericValues: IdColumna → double in metres for range-option rows where the user typed a value.
        var numericValues = CascadeRows
            .Where(r => r.SelectedOptionIsRange && r.RangeInput.HasValue)
            .ToDictionary(r => r.IdColumna, r => ToMetres(r.RangeInput!.Value, r.RangeUnit));

        return new GeneratorInput(
            codigo,
            capituloTitle,
            subcapituloTitle,
            seccionTitle,
            descripcion,
            coveninValues,
            selectedValores,
            numericValues);
    }

    /// <summary>
    /// Converts a user-typed numeric value to metres based on the column's unit.
    /// </summary>
    private static double ToMetres(double value, string? unit)
        => unit?.ToLowerInvariant() switch
        {
            "cm" => value / 100.0,
            "mm" => value / 1000.0,
            _    => value,   // assumed metres
        };

    // ── Material mapping refresh ──────────────────────────────────────────────

    /// <summary>
    /// Called when the MaterialMappingWindow is closed so Stage-2 dropdowns refresh.
    /// </summary>
    public void OnMaterialMappingClosed()
    {
        foreach (var row in CascadeRows.Where(r => r.IsMaterialColumn && r.SelectedOption != null))
            RefreshRevitMaterials(row);
    }
}
