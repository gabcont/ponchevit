using System.Windows;
using System.Windows.Controls;
using Ponchevit.Revit;
using Ponchevit.Ui.MaterialMapping;

namespace Ponchevit.Ui.AgregarFamilia;

/// <summary>
/// Code-behind for AgregarFamiliaWindow.
/// Responsibilities: route View events to the VM; open the modeless MaterialMappingWindow.
/// Fix #6: no RevitAPI imports here. Project materials are queried in the Command layer
/// (IProjectMaterialQuery) and passed into the VM via constructor. When "Mapear material"
/// is triggered from the VM, this code-behind opens the mapping window by re-querying
/// materials via IProjectMaterialQuery through IRevitContext.
/// </summary>
public partial class AgregarFamiliaWindow : Window
{
    private readonly AgregarFamiliaViewModel _vm;

    // Services needed to open the MaterialMappingWindow modeless on top.
    private readonly Data.ICoveninRulesRepository _rulesRepo;
    private readonly Data.IMaterialMappingRepository _materialRepo;
    private readonly Revit.Context.IRevitContext _revitContext;
    private readonly IProjectMaterialQuery _materialQuery;
    private readonly Infrastructure.ILog _log;

    public AgregarFamiliaWindow(
        AgregarFamiliaViewModel vm,
        Data.ICoveninRulesRepository rulesRepo,
        Data.IMaterialMappingRepository materialRepo,
        Revit.Context.IRevitContext revitContext,
        IProjectMaterialQuery materialQuery,
        Infrastructure.ILog log)
    {
        InitializeComponent();
        _vm            = vm;
        _rulesRepo     = rulesRepo;
        _materialRepo  = materialRepo;
        _revitContext  = revitContext;
        _materialQuery = materialQuery;
        _log           = log;

        DataContext = vm;

        vm.CloseRequested          += (_, _) => Close();
        vm.MapearMaterialRequested += (_, _) => OpenMaterialMappingWindow();
    }

    // ── Tree selection routing ─────────────────────────────────────────────

    /// <summary>
    /// Routes TreeViewItem selection to the VM command. Required because
    /// TreeViewItem.Selected is a routed event — binding SelectedItem on the
    /// TreeView alone does not fire the VM command per-node.
    /// </summary>
    private void TreeViewItem_Selected(object sender, RoutedEventArgs e)
    {
        if (sender is TreeViewItem { DataContext: TreeNodeViewModel node })
            _vm.SelectTreeNodeCommand.Execute(node);

        e.Handled = true; // Prevent bubbling to parent nodes.
    }

    // ── Partida selection routing ─────────────────────────────────────────

    /// <summary>
    /// Routes DataGrid row selection to the VM's SelectedPartidaItem property.
    /// Backfill is triggered separately by the "Seleccionar Partida" button (ConfirmPartidaCommand).
    /// </summary>
    private void PartidasDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (sender is DataGrid grid && grid.SelectedItem is PartidaDisplayItem item)
            _vm.SelectedPartidaItem = item;
    }

    // ── Material mapping sub-window ────────────────────────────────────────

    /// <summary>
    /// Opens the MaterialMappingWindow modeless (on top of AgregarFamiliaWindow).
    /// Fix #6 / Fix E: materials are fetched via IProjectMaterialQuery which obtains
    /// the Document internally — no RevitAPI type references in this code-behind.
    /// </summary>
    private void OpenMaterialMappingWindow()
    {
        var materials = _materialQuery.GetProjectMaterials();

        var mappingVm = new MaterialMappingViewModel(
            materials, _rulesRepo, _materialRepo, _revitContext, _log);

        var mappingWindow = new MaterialMappingWindow(mappingVm);
        mappingWindow.Closed += (_, _) => _vm.OnMaterialMappingClosed();
        mappingWindow.Show();
    }
}
