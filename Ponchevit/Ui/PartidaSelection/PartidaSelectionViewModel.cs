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
using Ponchevit.Domain.Matching;
using Ponchevit.Domain.Model;
using Ponchevit.Domain.Query;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.Context;
using Ponchevit.Revit.Families;
using Ponchevit.Ui.Common;
using Ponchevit.Ui.MaterialMapping;

namespace Ponchevit.Ui.PartidaSelection;

public sealed partial class PartidaSelectionViewModel : ObservableObject
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

    // ── Mode fields ──────────────────────────────────────────────────────────
    private readonly WindowMode _mode;
    private readonly Action<AssignInput>? _assignAction;
    private readonly ElementTopology? _targetTopology;
    private readonly Func<ElementTopology, PrefillResult>? _recognizeFunc;

    // ── State ────────────────────────────────────────────────────────────────
    private readonly CascadeMenuBuilder _cascadeBuilder;
    private readonly FilteredPartidaCollection _filteredPartidas;

    // Fix #2: track (IdConexion, CodigoAportado) pairs instead of just IDs.
    // CodigoAportado is cached from the MenuOption at selection time so
    // AssembleCodeFromPath never needs to re-query the repository.
    private readonly List<(string IdConexion, string CodigoAportado)> _selectedConnectionPath = new();

    // Number of cascade rows seeded from the left-tree selection (locked in UI).
    private int _seededRowCount;

    // Last PrefillResult from Reconocer, retained so AppendWithPrefill can re-apply it
    // when the user manually fills an Undetectable structural row.
    private PrefillResult? _lastPrefillResult;

    // ── Observable UI state ──────────────────────────────────────────────────
    public ObservableCollection<TreeNodeViewModel> TreeRoots { get; } = new();
    public ObservableCollection<CascadeRowViewModel> CascadeRows { get; } = new();

    /// <summary>
    /// Filtered + constructibility-annotated partidas for the right panel.
    /// Rebuilt whenever the underlying filter or cascade selection changes.
    /// </summary>
    public ObservableCollection<PartidaDisplayItem> PartidaDisplayItems { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPerformAction))]
    private TreeNodeViewModel? _selectedTreeNode;

    [ObservableProperty]
    private string _noRulesMessage = string.Empty;

    [ObservableProperty]
    private bool _hasNoRules;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PerformActionCommand))]
    private bool _canPerformAction;

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

    /// <summary>
    /// The partida explicitly confirmed by the user via "Seleccionar Partida".
    /// In Assign mode, this drives both code display and Asignar enablement.
    /// Cleared when the user selects a different tree node.
    /// </summary>
    [ObservableProperty]
    private PartidaDisplayItem? _confirmedPartidaItem;

    partial void OnConfirmedPartidaItemChanged(PartidaDisplayItem? value)
        => UpdateCanPerformAction();

    // Fix #9: user-facing status feedback after Agregar/Asignar.
    [ObservableProperty]
    private string? _statusMessage;

    [ObservableProperty]
    private bool _statusIsError;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPrefillReport))]
    private int _prefillAutoFilledCount;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPrefillReport))]
    private int _prefillAmbiguousCount;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPrefillReport))]
    private int _prefillUndetectableCount;

    /// <summary>
    /// Per-column detail lines for the prefill report strip (Assign mode only).
    /// Rebuilt whenever UpdatePrefillReport runs; cleared on tree node selection.
    /// </summary>
    public ObservableCollection<PrefillReportLine> PrefillReportLines { get; } = new();

    /// <summary>
    /// True once at least one recognition pass has produced results for the current element.
    /// Drives visibility of the prefill report strip.
    /// </summary>
    public bool HasPrefillReport =>
        IsAssignMode
        && (_prefillAutoFilledCount + _prefillAmbiguousCount + _prefillUndetectableCount) > 0;

    [ObservableProperty]
    private string? _targetElementDisplayName;

    // ── Computed properties ──────────────────────────────────────────────────
    public string WindowTitle       => _mode == WindowMode.Generate
        ? "Agregar Familia COVENIN" : "Asignar Código COVENIN";
    public string ActionButtonLabel => _mode == WindowMode.Generate
        ? "Agregar" : "Asignar";
    public bool IsAssignMode => _mode == WindowMode.Assign;

    public event EventHandler? CloseRequested;
    /// <summary>Raised when the "Mapear Material" link is clicked; subscribers open the mapping window.</summary>
    public event EventHandler? MapearMaterialRequested;

    // ── Constructor ──────────────────────────────────────────────────────────
    public PartidaSelectionViewModel(
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
        ILog log,
        WindowMode mode,
        string? targetElementDisplayName,
        Action<AssignInput>? assignAction,
        ElementTopology? targetTopology = null,
        Func<ElementTopology, PrefillResult>? recognizeFunc = null)
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

        _mode           = mode;
        _assignAction   = assignAction;
        _targetTopology = targetTopology;
        _recognizeFunc  = recognizeFunc;
        if (targetElementDisplayName != null)
            TargetElementDisplayName = targetElementDisplayName;

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
                    bool enabled = _mode == WindowMode.Assign
                        || _generators.Any(g => g.IsGenerable(sec.Codigo));
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
        ConfirmedPartidaItem = null;
        _lastPrefillResult = null;
        PrefillReportLines.Clear();

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
        UpdateCanPerformAction();
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

        changedRow.PrefillHint = null;

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

        AppendWithPrefill(changedRow.SelectedOption.IdConexion);

        UpdateCodeDisplay();
        UpdateRightPanelFromPath();
        RebuildDisplayItems();
        UpdateCanPerformAction();
        UpdatePrefillReport();
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

    /// <summary>
    /// Like AppendNextCascadeLevel, but if a PrefillResult is stored it chains through
    /// AutoFilled rows automatically, stopping at the first Ambiguous/Undetectable row
    /// (which is added to the UI so the user can fill it manually).
    /// CascadeRows.Add is always called AFTER SetSelectedSilently so that the
    /// PropertyChanged subscriber's IndexOf check returns -1 and is a no-op during
    /// the loop — same invariant as ApplyPrefillToCascade.
    /// </summary>
    private void AppendWithPrefill(string? parentConnectionId)
    {
        if (_lastPrefillResult == null || _mode != WindowMode.Assign)
        {
            AppendNextCascadeLevel(parentConnectionId);
            return;
        }

        bool continueTraversal = true;
        while (continueTraversal)
        {
            var level = _cascadeBuilder.GetNextLevel(parentConnectionId);
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

            if (_lastPrefillResult.TryGet(level.Columna.IdColumna, out var entry))
            {
                rowVm.PrefillHint = entry.State;

                if (entry.State == PrefillState.AutoFilled
                    && (entry.SuggestedIdValor != null || entry.SuggestedIdConexion != null))
                {
                    var option = FindMatchingOption(level.Options, entry);

                    if (option != null)
                    {
                        // SetSelectedSilently fires PropertyChanged → OnCascadeSelectionChanged,
                        // but rowVm is not yet in CascadeRows so IndexOf returns -1 → no-op.
                        rowVm.SetSelectedSilently(option);
                        _selectedConnectionPath.Add((option.IdConexion, option.CodigoAportado ?? string.Empty));
                        if (isMaterial) RefreshRevitMaterials(rowVm);
                        parentConnectionId = option.IdConexion;
                    }
                    else
                    {
                        rowVm.PrefillHint = PrefillState.Undetectable;
                        continueTraversal = false;
                    }
                }
                else
                {
                    continueTraversal = false;
                }
            }
            else
            {
                rowVm.PrefillHint = PrefillState.Undetectable;
                continueTraversal = false;
            }

            CascadeRows.Add(rowVm);
        }
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
        // The prefix mask can filter the partidas list even when no tree node
        // has been selected yet (e.g. the user clicked Reconocer immediately).
        string? secCodigo = SelectedTreeNode?.Kind == TreeNodeKind.Seccion     ? SelectedTreeNode.Codigo : null;
        string? subCodigo = SelectedTreeNode?.Kind == TreeNodeKind.Subcapitulo ? SelectedTreeNode.Codigo : null;
        string? capCodigo = SelectedTreeNode?.Kind == TreeNodeKind.Capitulo    ? SelectedTreeNode.Codigo : null;

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
            bool canBeConstructed = _mode == WindowMode.Assign
                || _constructibilityResolver.CanBeConstructed(p);
            PartidaDisplayItems.Add(new PartidaDisplayItem(p, canBeConstructed));
        }
    }

    // ── Partida confirmation — backfill cascade ───────────────────────────────

    [RelayCommand(CanExecute = nameof(CanConfirmPartida))]
    public void ConfirmPartida()
    {
        if (SelectedPartidaItem == null) return;
        if (_mode == WindowMode.Generate && !SelectedPartidaItem.CanBeConstructed) return;

        // In Assign mode, confirm the partida first — cascade backfill is best-effort.
        if (_mode == WindowMode.Assign)
            ConfirmedPartidaItem = SelectedPartidaItem;

        var path = _constructibilityResolver.GetPath(SelectedPartidaItem.CodigoPartida);

        if (path != null && path.Count > 0)
        {
            BackfillCascadeFromPath(path);
            UpdateCodeDisplay();
        }
        else if (_mode == WindowMode.Generate)
        {
            return;  // Generate still requires a valid constructible path
        }
        else
        {
            // Assign mode, no DAG rules: show the partida code directly in the code display
            string code = SelectedPartidaItem.CodigoPartida;
            CurrentCodeDisplay = code.Length >= 10 ? code[..10] : code.PadRight(10, 'X');
        }

        _filteredPartidas.ApplyFilter(codePrefix: SelectedPartidaItem.CodigoPartida);
        RebuildDisplayItems();
        UpdateCanPerformAction();
    }

    private bool CanConfirmPartida() =>
        _mode == WindowMode.Assign
            ? SelectedPartidaItem != null
            : SelectedPartidaItem?.CanBeConstructed == true;

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

    // ── Action enablement ─────────────────────────────────────────────────────

    private void UpdateCanPerformAction()
    {
        if (_mode == WindowMode.Assign)
        {
            CanPerformAction = _assignAction != null
                && TargetElementDisplayName != null
                && ConfirmedPartidaItem != null;
            return;
        }

        // Generate mode logic — unchanged below this line
        if (_selectedConnectionPath.Count == 0)
        {
            CanPerformAction = false;
            return;
        }

        bool allRowsSelected = CascadeRows.All(r =>
            r.SelectedOption != null
            && (!r.SelectedOptionIsRange || r.RangeInput.HasValue));
        if (!allRowsSelected)
        {
            CanPerformAction = false;
            return;
        }

        string assembled = AssembleCodeFromPath();
        CanPerformAction = assembled.Length == 10;
    }

    private void UpdatePrefillReport()
    {
        int autoFilled   = 0;
        int ambiguous    = 0;
        int undetectable = 0;

        foreach (var row in CascadeRows)
        {
            switch (row.PrefillHint)
            {
                case PrefillState.AutoFilled:   autoFilled++;   break;
                case PrefillState.Ambiguous:    ambiguous++;    break;
                case PrefillState.Undetectable: undetectable++; break;
            }
        }

        PrefillAutoFilledCount   = autoFilled;
        PrefillAmbiguousCount    = ambiguous;
        PrefillUndetectableCount = undetectable;

        // Rebuild per-column report lines from the stored PrefillResult.
        PrefillReportLines.Clear();
        if (_lastPrefillResult != null)
        {
            foreach (var kvp in _lastPrefillResult.Entries)
            {
                string columnaName = _rulesRepo.GetColumna(kvp.Key)?.Nombre ?? kvp.Key;

                string? detectedValue = null;
                if (kvp.Value.State == PrefillState.AutoFilled && kvp.Value.SuggestedIdValor != null)
                    detectedValue = _rulesRepo.GetValor(kvp.Value.SuggestedIdValor)?.DescripcionUi
                                    ?? kvp.Value.SuggestedIdValor;

                PrefillReportLines.Add(new PrefillReportLine(columnaName, kvp.Value.State, detectedValue));
            }
        }
    }

    /// <summary>
    /// Finds a MenuOption that matches the PrefillEntry.
    /// Prefers matching by IdConexion (exact, unambiguous) when SuggestedIdConexion is set;
    /// falls back to matching by IdValor when only SuggestedIdValor is set.
    /// </summary>
    private static MenuOption? FindMatchingOption(IReadOnlyList<MenuOption> options, PrefillEntry entry)
    {
        if (entry.SuggestedIdConexion != null)
        {
            var byConn = options.FirstOrDefault(o =>
                string.Equals(o.IdConexion, entry.SuggestedIdConexion, StringComparison.OrdinalIgnoreCase));
            if (byConn != null) return byConn;
        }

        if (entry.SuggestedIdValor != null)
        {
            return options.FirstOrDefault(o =>
                string.Equals(o.IdValor, entry.SuggestedIdValor, StringComparison.OrdinalIgnoreCase));
        }

        return null;
    }

    // ── Action command ────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanPerformAction))]
    private void PerformAction()
    {
        if (_mode == WindowMode.Generate)
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

            var capturedInput        = input;
            var capturedGenerator    = generator;
            var capturedOrchestrator = _orchestrator;

            StatusMessage = null;
            StatusIsError = false;

            _revitContext.PostExternalEvent(doc =>
            {
                try
                {
                    capturedOrchestrator.Generate(doc, capturedGenerator, capturedInput);
                    Application.Current?.Dispatcher.Invoke(() =>
                    {
                        StatusIsError = false;
                        StatusMessage = "Familia creada correctamente.";
                    });
                }
                catch (Exception ex)
                {
                    _log.Error("Agregar Familia: generation failed.", ex);
                    var userMessage = ex.Message;
                    Application.Current?.Dispatcher.Invoke(() =>
                    {
                        StatusIsError = true;
                        StatusMessage = $"Error: {userMessage}";
                    });
                }
            });
        }
        else
        {
            var assignInput = BuildAssignInput();
            if (assignInput == null) return;

            StatusMessage = null;
            StatusIsError = false;

            _assignAction?.Invoke(assignInput);
        }
    }

    [RelayCommand]
    private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);

    [RelayCommand]
    private void OpenMapearMaterial(CascadeRowViewModel? row)
        => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);

    // ── Reconocer (Assign mode) ──────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanReconocer))]
    private void Reconocer()
    {
        if (_targetTopology == null || _recognizeFunc == null) return;
        var result = _recognizeFunc(_targetTopology);
        ApplyPrefillToCascade(result);
    }

    private bool CanReconocer() =>
        _mode == WindowMode.Assign
        && _targetTopology != null
        && _recognizeFunc != null;

    private void ApplyPrefillToCascade(PrefillResult result)
    {
        _lastPrefillResult = result;

        while (CascadeRows.Count > _seededRowCount)
            CascadeRows.RemoveAt(CascadeRows.Count - 1);
        while (_selectedConnectionPath.Count > _seededRowCount)
            _selectedConnectionPath.RemoveAt(_selectedConnectionPath.Count - 1);

        string? currentParent = _seededRowCount > 0
            ? _selectedConnectionPath[_seededRowCount - 1].IdConexion
            : null;

        bool continueTraversal = true;
        while (continueTraversal)
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

            if (result.TryGet(level.Columna.IdColumna, out var entry))
            {
                rowVm.PrefillHint = entry.State;

                if (entry.State == PrefillState.AutoFilled
                    && (entry.SuggestedIdValor != null || entry.SuggestedIdConexion != null))
                {
                    var option = FindMatchingOption(level.Options, entry);

                    if (option != null)
                    {
                        rowVm.SetSelectedSilently(option);
                        _selectedConnectionPath.Add((option.IdConexion, option.CodigoAportado ?? string.Empty));
                        if (isMaterial) RefreshRevitMaterials(rowVm);
                        currentParent = option.IdConexion;
                    }
                    else
                    {
                        rowVm.PrefillHint = PrefillState.Undetectable;
                        continueTraversal = false;
                    }
                }
                else
                {
                    continueTraversal = false;
                }
            }
            else
            {
                rowVm.PrefillHint = PrefillState.Undetectable;
                continueTraversal = false;
            }

            CascadeRows.Add(rowVm);
        }

        UpdateCodeDisplay();
        UpdateRightPanelFromPath();
        RebuildDisplayItems();
        UpdateCanPerformAction();
        UpdatePrefillReport();
    }

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

    private AssignInput? BuildAssignInput()
    {
        if (ConfirmedPartidaItem == null) return null;

        string assembledCode = ConfirmedPartidaItem.CodigoPartida;
        if (assembledCode.Length != 10) return null;

        var (capitulo, subcapitulo, seccion) = _hierarchyResolver.Resolve(assembledCode);

        return new AssignInput(
            new CodigoCovenin(assembledCode),
            capitulo?.Titulo    ?? string.Empty,
            subcapitulo?.Titulo ?? string.Empty,
            seccion?.Titulo     ?? string.Empty,
            ConfirmedPartidaItem.Descripcion);
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
