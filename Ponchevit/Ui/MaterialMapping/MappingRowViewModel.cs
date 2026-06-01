using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Ponchevit.Ui.MaterialMapping;

public partial class MappingRowViewModel : ObservableObject
{
    private readonly IReadOnlyList<CoveninValueOption> _options;

    public string RevitMaterialName { get; }

    /// <summary>Display text of the first substring-match suggestion. Null when no match.</summary>
    public string? Suggestion { get; }

    /// <summary>IdValor of the first suggestion, used by AcceptSuggestionCommand.</summary>
    public string? SuggestionId { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedCoveninDisplay))]
    [NotifyPropertyChangedFor(nameof(IsDirty))]
    private string? _selectedCoveninValueId;

    private readonly string? _originalCoveninValueId;

    public bool IsDirty => SelectedCoveninValueId != _originalCoveninValueId;

    /// <summary>Human-readable label for the current selection, shown in the non-edit cell.</summary>
    public string SelectedCoveninDisplay
        => _options.FirstOrDefault(o => o.IdValor == SelectedCoveninValueId)?.Display
           ?? "(Sin mapear)";

    public MappingRowViewModel(
        string revitMaterialName,
        string? suggestionId,
        string? suggestionDisplay,
        string? currentMapping,
        IReadOnlyList<CoveninValueOption> options)
    {
        RevitMaterialName = revitMaterialName;
        SuggestionId = suggestionId;
        Suggestion = suggestionDisplay;
        _selectedCoveninValueId = currentMapping;
        _originalCoveninValueId = currentMapping;
        _options = options;
    }

    [RelayCommand(CanExecute = nameof(CanAcceptSuggestion))]
    private void AcceptSuggestion() => SelectedCoveninValueId = SuggestionId;

    private bool CanAcceptSuggestion() => SuggestionId != null;
}
