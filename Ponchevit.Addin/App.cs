using Autodesk.Revit.UI;
using Microsoft.Extensions.DependencyInjection;
using Ponchevit.Core;
using Ponchevit.Presentation.Views;
using System.IO;
using System;
using System.Reflection;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Markup;

namespace Ponchevit.Addin
{
    public class App : IExternalApplication
    {
        public static IServiceProvider ServiceProvider { get; private set; }
        public static ExternalEvent Event { get; private set; }

        public Result OnStartup(UIControlledApplication application)
        {
            // 1. Configurar DI
            var services = new ServiceCollection();
            var handler = new CoveninExternalEventHandler();
            Event = ExternalEvent.Create(handler);
            
            services.AddSingleton(handler);
            services.AddSingleton<AgregarCodigoView>();
            // Aquí registrará sus ViewModels más adelante
            ServiceProvider = services.BuildServiceProvider();

            RegisterRibbon(application);

            return Result.Succeeded;
        }

      private static void RegisterRibbon(UIControlledApplication application)
{
    const string tabName = "Ponchevit";
    const string panelName = "Codificación";

    try { application.CreateRibbonTab(tabName); } catch { }

    RibbonPanel panel = application.CreateRibbonPanel(tabName, panelName);
    string assemblyPath = Assembly.GetExecutingAssembly().Location;

    // 1. Agregar Familia (+)
    PushButtonData agregarFamiliaButton = new PushButtonData(
        "btnAgregarFamilia", "Agregar\nFamilia", assemblyPath, "Ponchevit.Addin.Commands.AgregarFamiliaCommand");
    agregarFamiliaButton.LargeImage = GetImage("Ponchevit.Addin.Resources.Add.xaml");
    agregarFamiliaButton.Image = GetImage("Ponchevit.Addin.Resources.Add.xaml");

    // 2. Asignar Código (<>)
    PushButtonData asignarCodigoButton = new PushButtonData(
        "btnAsignarCodigo", "Asignar\nCódigo", assemblyPath, "Ponchevit.Addin.Commands.AsignarCodigoCommand");
    asignarCodigoButton.LargeImage = GetImage("Ponchevit.Addin.Resources.Code.xaml");
    asignarCodigoButton.Image = GetImage("Ponchevit.Addin.Resources.Code.xaml");

    // 3. Generar Automático (Rayo)
    PushButtonData generarAutomaticoButton = new PushButtonData(
        "btnGenerarAutomatico", "Generar\nAutomático", assemblyPath, "Ponchevit.Addin.Commands.GenerarAutomaticoCommand");
    generarAutomaticoButton.LargeImage = GetImage("Ponchevit.Addin.Resources.Lightning.xaml");
    generarAutomaticoButton.Image = GetImage("Ponchevit.Addin.Resources.Lightning.xaml");

    panel.AddItem(agregarFamiliaButton);
    panel.AddItem(asignarCodigoButton);
    panel.AddItem(generarAutomaticoButton);
}

        private static ImageSource? GetImage(string resourceName)
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            using Stream? stream = assembly.GetManifestResourceStream(resourceName);

            if (stream == null)
            {
                return null;
            }

            object resource = XamlReader.Load(stream);
            if (resource is DrawingImage drawingImage)
            {
                drawingImage.Freeze();
                return drawingImage;
            }

            if (resource is ImageSource imageSource)
            {
                if (imageSource.CanFreeze)
                {
                    imageSource.Freeze();
                }

                return imageSource;
            }

            return null;
        }

        public Result OnShutdown(UIControlledApplication application) => Result.Succeeded;
    }
}