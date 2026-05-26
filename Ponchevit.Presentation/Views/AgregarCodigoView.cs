using System.Windows;

namespace Ponchevit.Presentation.Views;

public class AgregarCodigoView : Window
{
    public AgregarCodigoView()
    {
        Title = "Agregar Código";
        Width = 480;
        Height = 320;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Content = new System.Windows.Controls.TextBlock
        {
            Text = "Vista inicial de Agregar Código",
            Margin = new Thickness(24),
            VerticalAlignment = VerticalAlignment.Center,
            HorizontalAlignment = HorizontalAlignment.Center
        };
    }
}