using Autodesk.Revit.DB;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.UI;
using Autodesk.Revit.UI.Selection;
using System;
using System.IO;
using System.Reflection;
using System.Windows.Media.Imaging;
using Ponchevit.Composition;
using Ponchevit.Infrastructure;

namespace Ponchevit
{
    public class App : IExternalApplication
    {
        internal static Services Services { get; private set; } = null!;

        public Result OnStartup(UIControlledApplication application)
        {
            Services = Services.Build();
            Services.Log.Info("Ponchevit OnStartup");

            string tabName = "Ponchevit USM";

            // Create a custom ribbon tab (ignore error if already exists)
            try
            {
                application.CreateRibbonTab(tabName);
            }
            catch (Exception)
            {
                // If the tab already exists, you can ignore or log the exception.
            }

            // Create a panel on the custom tab
            RibbonPanel panel = application.CreateRibbonPanel(tabName, "Acciones");

            // Get the path to this assembly
            string assemblyPath = Assembly.GetExecutingAssembly().Location;

            // Phase 3.5 — Mapeo de Materiales
            PushButtonData btnMapeo = new PushButtonData(
                "MapeoMateriales",
                "Mapeo de\nMateriales",
                assemblyPath,
                "Ponchevit.Commands.MapeoMaterialesCommand");
            BitmapImage icon = LoadIcon();

            btnMapeo.ToolTip = "Mapea materiales de Revit a valores COVENIN para la codificación.";
            btnMapeo.Image = icon;
            btnMapeo.LargeImage = icon;
            panel.AddItem(btnMapeo);

            // Phase 4 — Agregar Familia
            PushButtonData btnAgregar = new PushButtonData(
                "AgregarFamilia",
                "Agregar\nFamilia",
                assemblyPath,
                "Ponchevit.Commands.AgregarFamiliaCommand");
            btnAgregar.ToolTip = "Crea una familia COVENIN en el proyecto activo.";
            btnAgregar.Image = icon;
            btnAgregar.LargeImage = icon;
            panel.AddItem(btnAgregar);

            // Phase 5.7 — Asignar Código
            PushButtonData btnAsignar = new PushButtonData(
                "AsignarCodigo",
                "Asignar\nCódigo",
                assemblyPath,
                "Ponchevit.Commands.AsignarCodigoCommand");
            btnAsignar.ToolTip = "Asigna un código COVENIN al elemento seleccionado. Seleccione un elemento antes de hacer clic.";
            btnAsignar.Image = icon;
            btnAsignar.LargeImage = icon;
            panel.AddItem(btnAsignar);

            // Phase 6.7 — Codificación Dashboard
            PushButtonData btnDashboard = new PushButtonData(
                "CodificacionDashboard",
                "Codificación\nDashboard",
                assemblyPath,
                "Ponchevit.Commands.CodificacionDashboardCommand");
            btnDashboard.ToolTip = "Muestra el inventario de familias del proyecto con su estado de codificación COVENIN.";
            btnDashboard.Image = icon;
            btnDashboard.LargeImage = icon;
            panel.AddItem(btnDashboard);

            return Result.Succeeded;
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            Services?.Log.Info("Ponchevit OnShutdown");
            return Result.Succeeded;
        }

        private static BitmapImage LoadIcon()
        {
            string assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
            string iconPath = Path.Combine(assemblyDir, "Assets", "icon.png");
            return new BitmapImage(new Uri(iconPath));
        }
    }
}
