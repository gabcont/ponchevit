using System.Windows;

namespace Ponchevit.Ui.Codificacion;

public partial class CodificacionDashboardWindow : Window
{
    public CodificacionDashboardWindow(CodificacionDashboardViewModel vm)
    {
        InitializeComponent();
        DataContext = vm;
        vm.CloseRequested += (_, _) => Close();
    }
}
