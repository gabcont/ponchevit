using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Ponchevit.Data;
using Ponchevit.Domain.Materials;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.Context;

namespace Ponchevit.Ui.MaterialMapping;

/// <summary>Option item for the Covenin value dropdown. IdValor = null → "(Sin mapear)".</summary>
public sealed record CoveninValueOption(string? IdValor, string Display);

public partial class MaterialMappingViewModel : ObservableObject
{
    private readonly IMaterialMappingRepository _repo;
    private readonly IRevitContext _revitContext;
    private readonly ILog _log;

    public ObservableCollection<MappingRowViewModel> Rows { get; }

    /// <summary>
    /// Shared option list for every row's ComboBox.
    /// First item is always "(Sin mapear)" with IdValor = null.
    /// Remaining items are Covenin Valor entries from material-type columns.
    /// </summary>
    public IReadOnlyList<CoveninValueOption> CoveninOptions { get; }

    public event EventHandler? CloseRequested;

    public MaterialMappingViewModel(
        IReadOnlyList<string> revitMaterials,
        ICoveninRulesRepository rulesRepo,
        IMaterialMappingRepository mappingRepo,
        IRevitContext revitContext,
        ILog log)
    {
        _repo = mappingRepo;
        _revitContext = revitContext;
        _log = log;

        // Identify material-type columns by name; fallback to all values if none found.
        var materialColumnIds = rulesRepo.GetColumnas()
            .Where(c => c.Nombre.Contains("MATERIAL", StringComparison.OrdinalIgnoreCase))
            .Select(c => c.IdColumna)
            .ToHashSet();

        var materialValues = (materialColumnIds.Count > 0
            ? rulesRepo.GetValores().Where(v => materialColumnIds.Contains(v.IdColumna))
            : rulesRepo.GetValores())
            .OrderBy(v => v.DescripcionUi)
            .ToList();

        CoveninOptions = new[] { new CoveninValueOption(null, "(Sin mapear)") }
            .Concat(materialValues.Select(v => new CoveninValueOption(v.IdValor, v.DescripcionUi)))
            .ToList();

        var currentMapping = mappingRepo.GetAll();

        Rows = new ObservableCollection<MappingRowViewModel>(
            revitMaterials.Select(name =>
            {
                currentMapping.TryGetValue(name, out var currentId);

                var suggestions = SubstringSuggester.Suggest(name, materialValues);
                var firstId      = suggestions.FirstOrDefault();
                var firstDisplay = firstId != null
                    ? materialValues.FirstOrDefault(v => v.IdValor == firstId)?.DescripcionUi
                    : null;

                return new MappingRowViewModel(name, firstId, firstDisplay, currentId, CoveninOptions);
            }));
    }

    /// <summary>
    /// Posts all dirty rows to the repository via ExternalEvent then closes the window.
    /// The write happens on Revit's main thread; the window closes immediately.
    /// </summary>
    [RelayCommand]
    private void Save()
    {
        var dirty = Rows.Where(r => r.IsDirty).ToList();

        if (dirty.Count > 0)
        {
            _revitContext.PostExternalEvent(_ =>
            {
                foreach (var row in dirty)
                {
                    if (row.SelectedCoveninValueId == null)
                        _repo.Remove(row.RevitMaterialName);
                    else
                        _repo.Set(row.RevitMaterialName, row.SelectedCoveninValueId);
                }
                _log.Info($"Mapeo de Materiales: {dirty.Count} row(s) updated.");
            });
        }

        CloseRequested?.Invoke(this, EventArgs.Empty);
    }

    [RelayCommand]
    private void Cancel() => CloseRequested?.Invoke(this, EventArgs.Empty);
}
