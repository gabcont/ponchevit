using System.Windows;
using System.Windows.Controls;
using Ponchevit.Revit;
using Ponchevit.Ui.MaterialMapping;

namespace Ponchevit.Ui.PartidaSelection;

public partial class PartidaSelectionWindow : Window
{
    private readonly PartidaSelectionViewModel _vm;

    // Services needed to open the MaterialMappingWindow modeless on top.
    private readonly Data.ICoveninRulesRepository _rulesRepo;
    private readonly Data.IMaterialMappingRepository _materialRepo;
    private readonly Revit.Context.IRevitContext _revitContext;
    private readonly IProjectMaterialQuery _materialQuery;
    private readonly Infrastructure.ILog _log;

    public PartidaSelectionWindow(
        PartidaSelectionViewModel vm,
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

    private void TreeViewItem_Selected(object sender, RoutedEventArgs e)
    {
        if (sender is TreeViewItem { DataContext: TreeNodeViewModel node })
            _vm.SelectTreeNodeCommand.Execute(node);
        e.Handled = true;
    }

    private void PartidasDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (sender is DataGrid grid && grid.SelectedItem is PartidaDisplayItem item)
            _vm.SelectedPartidaItem = item;
    }

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
