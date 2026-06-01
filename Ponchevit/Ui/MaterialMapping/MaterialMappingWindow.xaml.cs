using System.Windows;

namespace Ponchevit.Ui.MaterialMapping;

public partial class MaterialMappingWindow : Window
{
    public MaterialMappingWindow(MaterialMappingViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
        vm.CloseRequested += (_, _) => Close();
    }
}
