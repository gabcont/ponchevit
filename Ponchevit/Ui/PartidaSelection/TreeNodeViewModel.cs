using System.Collections.Generic;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Ponchevit.Ui.PartidaSelection;

/// <summary>
/// Represents one node in the left-tree (Capítulo / Subcapítulo / Sección).
/// A node is enabled only when its corresponding Sección has a registered
/// IFamilyGenerator. Disabled nodes render greyed-out but remain visible.
/// </summary>
public sealed partial class TreeNodeViewModel : ObservableObject
{
    public string Label { get; }
    public string? Codigo { get; }
    public TreeNodeKind Kind { get; }

    /// <summary>
    /// False → node is greyed out in the UI. Leaf sección nodes are only enabled
    /// when a generator is registered for their category.
    /// </summary>
    public bool IsEnabled { get; }

    public ObservableCollection<TreeNodeViewModel> Children { get; } = new();

    [ObservableProperty]
    private bool _isExpanded;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSelectedAndEnabled))]
    private bool _isSelected;

    public bool IsSelectedAndEnabled => IsSelected && IsEnabled;

    public TreeNodeViewModel(string label, string? codigo, TreeNodeKind kind, bool isEnabled)
    {
        Label     = label;
        Codigo    = codigo;
        Kind      = kind;
        IsEnabled = isEnabled;
    }
}

public enum TreeNodeKind
{
    Capitulo,
    Subcapitulo,
    Seccion,
}
