using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using Ponchevit.Domain.Matching;
using Ponchevit.Domain.Query;

namespace Ponchevit.Ui.PartidaSelection;

/// <summary>
/// Represents one level in the central-panel cascading dropdown.
/// The ColumnName is the COVENIN column label shown as a row header.
/// Options are the selectable values for this level.
/// IsMaterialColumn indicates the two-stage material control should be shown.
/// IsSeeded indicates the row was pre-selected from the left-tree — it renders
/// highlighted to show the user the value was inferred from their tree selection.
/// SelectedOptionIsRange is true when the chosen option is a true range (NumMin ≠ NumMax);
/// in that case the UI shows an additional RangeInput TextBox below the ComboBox.
/// </summary>
public sealed partial class CascadeRowViewModel : ObservableObject
{
    public string ColumnName { get; }
    public string IdColumna { get; }
    public bool IsMaterialColumn { get; }

    /// <summary>
    /// True when this row was pre-selected from the left-tree node selection.
    /// Seeded rows are visually highlighted in the UI.
    /// </summary>
    public bool IsSeeded { get; }

    /// <summary>
    /// True when all options in this column carry numeric range data (NumMin/NumMax).
    /// Retained for internal tracking only — does NOT control ComboBox visibility.
    /// </summary>
    public bool IsNumericColumn { get; }

    /// <summary>
    /// Set by the VM after recognition. Null means no prefill hint or the user has
    /// manually overridden the auto-filled value.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPrefillAutoFilled))]
    [NotifyPropertyChangedFor(nameof(IsPrefillAmbiguous))]
    [NotifyPropertyChangedFor(nameof(IsPrefillUndetectable))]
    private PrefillState? _prefillHint;

    public bool IsPrefillAutoFilled   => _prefillHint == PrefillState.AutoFilled;
    public bool IsPrefillAmbiguous    => _prefillHint == PrefillState.Ambiguous;
    public bool IsPrefillUndetectable => _prefillHint == PrefillState.Undetectable;

    public ObservableCollection<MenuOption> Options { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedIdConexion))]
    [NotifyPropertyChangedFor(nameof(SelectedIdValor))]
    [NotifyPropertyChangedFor(nameof(SelectedOptionIsRange))]
    [NotifyPropertyChangedFor(nameof(RangeMin))]
    [NotifyPropertyChangedFor(nameof(RangeMax))]
    [NotifyPropertyChangedFor(nameof(RangeUnit))]
    private MenuOption? _selectedOption;

    /// <summary>
    /// True when the currently selected option is a true range (NumMin ≠ NumMax).
    /// When true, the UI shows an additional TextBox for a concrete value.
    /// </summary>
    public bool SelectedOptionIsRange =>
        SelectedOption?.NumMin.HasValue == true
        && SelectedOption?.NumMax.HasValue == true
        && SelectedOption.NumMin!.Value != SelectedOption.NumMax!.Value;

    /// <summary>Min bound of the selected range option (0 when not a range).</summary>
    public double RangeMin => SelectedOption?.NumMin ?? 0.0;

    /// <summary>Max bound of the selected range option (0 when not a range).</summary>
    public double RangeMax => SelectedOption?.NumMax ?? 0.0;

    /// <summary>Unit label from the selected option's ValorData (e.g. "cm", "mm").</summary>
    public string? RangeUnit => SelectedOption?.ValorData?.Unidad;

    /// <summary>The user-typed value for a range option (null when no range option is selected).</summary>
    [ObservableProperty]
    private double? _rangeInput;

    public string? SelectedIdConexion => SelectedOption?.IdConexion;
    public string? SelectedIdValor   => SelectedOption?.IdValor;

    // ── Material two-stage support ──────────────────────────────────────────
    public ObservableCollection<RevitMaterialOption> RevitMaterialOptions { get; } = new();

    [ObservableProperty]
    private RevitMaterialOption? _selectedRevitMaterial;

    [ObservableProperty]
    private bool _showMapearMaterialLink;

    public event EventHandler? MapearMaterialRequested;

    // Tracks whether we are in a silent-set operation so PropertyChanged is suppressed.
    private bool _suppressPropertyChanged;

    public CascadeRowViewModel(
        string columnName,
        string idColumna,
        IEnumerable<MenuOption> options,
        bool isMaterialColumn,
        bool isSeeded = false,
        bool isNumericColumn = false)
    {
        ColumnName       = columnName;
        IdColumna        = idColumna;
        IsMaterialColumn = isMaterialColumn;
        IsSeeded         = isSeeded;
        IsNumericColumn  = isNumericColumn;
        Options          = new ObservableCollection<MenuOption>(options);
    }

    /// <summary>
    /// Sets SelectedOption without triggering the cascade rebuilds that the normal
    /// PropertyChanged handler causes (used during seeding / backfill).
    /// After this call the row reports the correct SelectedOption; bindings update via
    /// explicit OnPropertyChanged calls, but the external PropertyChanged subscriber
    /// that fires cascade rebuilds will not see a notification.
    /// </summary>
    public void SetSelectedSilently(MenuOption option)
    {
        _suppressPropertyChanged = true;
        try
        {
            // Write through the generated property so the CommunityToolkit
            // [ObservableProperty] bookkeeping stays consistent, but our overridden
            // OnPropertyChanged drops the notification while the flag is set.
            SelectedOption = option;
        }
        finally
        {
            _suppressPropertyChanged = false;
        }
        // Manually notify bindings after the flag is cleared so the UI reflects the value.
        OnPropertyChanged(nameof(SelectedOption));
        OnPropertyChanged(nameof(SelectedIdConexion));
        OnPropertyChanged(nameof(SelectedIdValor));
        OnPropertyChanged(nameof(SelectedOptionIsRange));
        OnPropertyChanged(nameof(RangeMin));
        OnPropertyChanged(nameof(RangeMax));
        OnPropertyChanged(nameof(RangeUnit));
    }

    // Override to suppress notifications during silent-set.
    protected override void OnPropertyChanging(System.ComponentModel.PropertyChangingEventArgs e)
    {
        if (!_suppressPropertyChanged)
            base.OnPropertyChanging(e);
    }

    // CommunityToolkit source-generator calls OnPropertyChanged after each set.
    // We intercept by checking the suppress flag.
    protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (_suppressPropertyChanged) return;
        base.OnPropertyChanged(e);
    }

    /// <summary>
    /// Called by the VM when the Revit-material list for the current Covenin selection
    /// has been refreshed.
    /// </summary>
    public void SetRevitMaterials(IEnumerable<RevitMaterialOption> materials)
    {
        RevitMaterialOptions.Clear();
        foreach (var m in materials)
            RevitMaterialOptions.Add(m);

        ShowMapearMaterialLink = RevitMaterialOptions.Count == 0 && SelectedOption != null;
        SelectedRevitMaterial  = RevitMaterialOptions.FirstOrDefault();
    }

    public void RaiseMapearMaterial() => MapearMaterialRequested?.Invoke(this, EventArgs.Empty);
}

/// <summary>An option in the inner (Revit-material) dropdown.</summary>
public sealed record RevitMaterialOption(string RevitMaterialName, string Display);
