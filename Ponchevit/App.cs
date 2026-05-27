using Autodesk.Revit.DB;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.UI;
using Autodesk.Revit.UI.Selection;
using System;
using System.Reflection;
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

            // Add Command One button
            PushButtonData btnData1 = new PushButtonData(
                "CommandOne",
                "Command One",
                assemblyPath,
                "Ponchevit.Commands.CommandOne");
            btnData1.ToolTip = "Executes Command One.";
            panel.AddItem(btnData1);


            return Result.Succeeded;
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            Services?.Log.Info("Ponchevit OnShutdown");
            return Result.Succeeded;
        }
    }
}
