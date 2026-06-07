using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Ponchevit.Data;
using Ponchevit.Domain.Catalog;
using Ponchevit.Domain.Codificacion;
using Ponchevit.Domain.Materials;
using Ponchevit.Domain.Matching;
using Ponchevit.Infrastructure;
using Ponchevit.Revit;
using Ponchevit.Revit.Codificacion;
using Ponchevit.Revit.Context;
using Ponchevit.Revit.Families;
using Ponchevit.Ui.PartidaSelection;

namespace Ponchevit.Ui.Codificacion;

public sealed partial class CodificacionDashboardViewModel : ObservableObject
{
    // ── Injected services ────────────────────────────────────────────────────
    private readonly IPartidasRepository _partidasRepo;
    private readonly ICoveninRulesRepository _coveninRulesRepo;
    private readonly PartidaCatalog _catalog;
    private readonly PartidaConstructibilityResolver _constructibilityResolver;
    private readonly IMaterialMappingResolver _materialResolver;
    private readonly IMaterialMappingRepository _materialMappingRepo;
    private readonly IRevitContext _revitContext;
    private readonly IFamilyGenerator[] _generators;
    private readonly FamilyGenerationOrchestrator _generationOrchestrator;
    private readonly PartidaHierarchyResolver _hierarchyResolver;
    private readonly AssignCodeOrchestrator _assignCodeOrchestrator;
    private readonly IProjectMaterialQuery _projectMaterialQuery;
    private readonly ProjectInventoryReader _inventoryReader;
    private readonly CodificacionScheduleBuilder _scheduleBuilder;
    private readonly Func<ElementTopology, PrefillResult> _recognizeTopology;
    private readonly Func<ElementTopology, bool> _canRecognizeTopology;
    private readonly ILog _log;

    private IReadOnlyList<string> _projectMaterials;
    private List<CodificacionRowViewModel> _allRows = new();

    // ── Observable state ─────────────────────────────────────────────────────
    [ObservableProperty]
    private string _searchText = string.Empty;

    [ObservableProperty]
    private StatusFilter _activeFilter = StatusFilter.Todas;

    [ObservableProperty]
    private string _aggregateHeader = string.Empty;

    [ObservableProperty]
    private bool _hasRows;

    [ObservableProperty]
    private string? _statusMessage;

    [ObservableProperty]
    private bool _statusIsError;

    public ObservableCollection<CodificacionRowViewModel> DisplayRows { get; } = new();

    public event EventHandler? CloseRequested;

    // ── Constructor ──────────────────────────────────────────────────────────
    public CodificacionDashboardViewModel(
        IReadOnlyList<CodificacionSummary> initialSummaries,
        IReadOnlyList<string> projectMaterials,
        IPartidasRepository partidasRepo,
        ICoveninRulesRepository coveninRulesRepo,
        PartidaCatalog catalog,
        PartidaConstructibilityResolver constructibilityResolver,
        IMaterialMappingResolver materialResolver,
        IMaterialMappingRepository materialMappingRepo,
        IRevitContext revitContext,
        IFamilyGenerator[] generators,
        FamilyGenerationOrchestrator generationOrchestrator,
        PartidaHierarchyResolver hierarchyResolver,
        AssignCodeOrchestrator assignCodeOrchestrator,
        IProjectMaterialQuery projectMaterialQuery,
        ProjectInventoryReader inventoryReader,
        CodificacionScheduleBuilder scheduleBuilder,
        Func<ElementTopology, PrefillResult> recognizeTopology,
        Func<ElementTopology, bool> canRecognizeTopology,
        ILog log)
    {
        _projectMaterials         = projectMaterials         ?? Array.Empty<string>();
        _partidasRepo             = partidasRepo             ?? throw new ArgumentNullException(nameof(partidasRepo));
        _coveninRulesRepo         = coveninRulesRepo         ?? throw new ArgumentNullException(nameof(coveninRulesRepo));
        _catalog                  = catalog                  ?? throw new ArgumentNullException(nameof(catalog));
        _constructibilityResolver = constructibilityResolver ?? throw new ArgumentNullException(nameof(constructibilityResolver));
        _materialResolver         = materialResolver         ?? throw new ArgumentNullException(nameof(materialResolver));
        _materialMappingRepo      = materialMappingRepo      ?? throw new ArgumentNullException(nameof(materialMappingRepo));
        _revitContext             = revitContext             ?? throw new ArgumentNullException(nameof(revitContext));
        _generators               = generators               ?? Array.Empty<IFamilyGenerator>();
        _generationOrchestrator   = generationOrchestrator   ?? throw new ArgumentNullException(nameof(generationOrchestrator));
        _hierarchyResolver        = hierarchyResolver        ?? throw new ArgumentNullException(nameof(hierarchyResolver));
        _assignCodeOrchestrator   = assignCodeOrchestrator   ?? throw new ArgumentNullException(nameof(assignCodeOrchestrator));
        _projectMaterialQuery     = projectMaterialQuery     ?? throw new ArgumentNullException(nameof(projectMaterialQuery));
        _inventoryReader          = inventoryReader          ?? throw new ArgumentNullException(nameof(inventoryReader));
        _scheduleBuilder          = scheduleBuilder          ?? throw new ArgumentNullException(nameof(scheduleBuilder));
        _recognizeTopology        = recognizeTopology        ?? throw new ArgumentNullException(nameof(recognizeTopology));
        _canRecognizeTopology     = canRecognizeTopology     ?? throw new ArgumentNullException(nameof(canRecognizeTopology));
        _log                      = log                      ?? throw new ArgumentNullException(nameof(log));

        LoadSummaries(initialSummaries ?? Array.Empty<CodificacionSummary>());
    }

    // ── Partial hooks ─────────────────────────────────────────────────────────
    partial void OnSearchTextChanged(string value) => ApplyFilter();
    partial void OnActiveFilterChanged(StatusFilter value) => ApplyFilter();

    // ── Data loading ──────────────────────────────────────────────────────────
    private void LoadSummaries(IReadOnlyList<CodificacionSummary> summaries)
    {
        _allRows = summaries.Select(s => new CodificacionRowViewModel(s)).ToList();
        ApplyFilter();
        UpdateAggregateHeader();
    }

    private void ApplyFilter()
    {
        var filtered = _allRows.AsEnumerable();

        if (!string.IsNullOrWhiteSpace(SearchText))
            filtered = filtered.Where(r =>
                r.FamilyTypeName.Contains(SearchText, StringComparison.OrdinalIgnoreCase));

        filtered = ActiveFilter switch
        {
            StatusFilter.Codificadas => filtered.Where(r => r.IsCodified),
            StatusFilter.SinCodigo   => filtered.Where(r => !r.IsCodified),
            _                        => filtered,
        };

        DisplayRows.Clear();
        foreach (var row in filtered)
            DisplayRows.Add(row);

        HasRows = DisplayRows.Count > 0;
    }

    private void UpdateAggregateHeader()
    {
        int total     = _allRows.Count;
        int codified  = _allRows.Count(r => r.IsCodified);

        if (total == 0)
        {
            AggregateHeader = "Sin familias en el modelo";
            return;
        }

        int pct = (int)Math.Round(codified * 100.0 / total, 0);
        AggregateHeader = $"{codified} de {total} familias codificadas ({pct}%)";
    }

    // ── Commands ──────────────────────────────────────────────────────────────
    [RelayCommand]
    private void Refrescar()
    {
        _revitContext.PostExternalEvent(doc =>
        {
            var summaries  = _inventoryReader.Read(doc);
            var materials  = _projectMaterialQuery.GetProjectMaterials();
            Application.Current?.Dispatcher.Invoke(() =>
            {
                _projectMaterials = materials;
                LoadSummaries(summaries);
            });
        });
    }

    [RelayCommand]
    private void GenerarSchedule()
    {
        StatusMessage = null;
        StatusIsError = false;

        _revitContext.PostExternalEvent(doc =>
        {
            try
            {
                string name = _scheduleBuilder.Build(doc);
                System.Windows.Application.Current?.Dispatcher.Invoke(() =>
                {
                    StatusIsError = false;
                    StatusMessage = $"Schedule creado: \"{name}\"";
                });
            }
            catch (Exception ex)
            {
                _log.Error("CodificacionDashboard: Generar Schedule failed.", ex);
                var msg = ex.Message;
                System.Windows.Application.Current?.Dispatcher.Invoke(() =>
                {
                    StatusIsError = true;
                    StatusMessage = $"Error: {msg}";
                });
            }
        });
    }

    [RelayCommand]
    private void AsignarCodigo(CodificacionRowViewModel row)
    {
        if (row == null) return;

        var capturedId = row.SampleInstanceId;
        _revitContext.PostExternalEvent(doc =>
        {
            var reader   = new ElementTopologyReader(_materialResolver);
            var topology = reader.ReadById(doc, capturedId);
            Application.Current?.Dispatcher.Invoke(() => OpenAssignWindow(row, topology));
        });
    }

    [RelayCommand]
    private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);

    // ── Assign window helper ──────────────────────────────────────────────────
    private void OpenAssignWindow(CodificacionRowViewModel row, ElementTopology? topology)
    {
        PartidaSelectionViewModel? vmRef = null;

        var capturedRow = row;

        Action<AssignInput> assignAction = input =>
            _revitContext.PostExternalEvent(doc =>
            {
                try
                {
                    _assignCodeOrchestrator.Assign(doc, capturedRow.SampleInstanceId, input);

                    Application.Current?.Dispatcher.Invoke(() =>
                    {
                        if (vmRef == null) return;
                        vmRef.StatusIsError = false;
                        vmRef.StatusMessage = "Código asignado correctamente.";
                    });
                }
                catch (Exception ex)
                {
                    _log.Error("CodificacionDashboard: Asignar Código write failed.", ex);
                    var msg = ex.Message;
                    Application.Current?.Dispatcher.Invoke(() =>
                    {
                        if (vmRef == null) return;
                        vmRef.StatusIsError = true;
                        vmRef.StatusMessage = $"Error: {msg}";
                    });
                }
            });

        var vm = new PartidaSelectionViewModel(
            _partidasRepo,
            _coveninRulesRepo,
            _catalog,
            _constructibilityResolver,
            _materialResolver,
            _materialMappingRepo,
            _revitContext,
            _generators,
            _generationOrchestrator,
            _projectMaterials,
            _hierarchyResolver,
            _log,
            mode: WindowMode.Assign,
            targetElementDisplayName: row.FamilyTypeName,
            assignAction: assignAction,
            targetTopology: topology,
            recognizeFunc: topology != null && _canRecognizeTopology(topology) ? _recognizeTopology : null);

        vmRef = vm;

        var window = new PartidaSelectionWindow(
            vm,
            _coveninRulesRepo,
            _materialMappingRepo,
            _revitContext,
            _projectMaterialQuery,
            _log);

        window.Show();
    }
}
