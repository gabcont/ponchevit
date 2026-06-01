using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Ponchevit.Ui.MaterialMapping;

namespace Ponchevit.Commands;

[Transaction(TransactionMode.Manual)]
public class MapeoMaterialesCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);

            var doc = commandData.Application.ActiveUIDocument.Document;
            var revitMaterials = GetProjectMaterials(doc);

            var vm = new MaterialMappingViewModel(
                revitMaterials,
                services.CoveninRulesRepository,
                services.MaterialMappingRepository,
                services.RevitContext,
                services.Log);

            var window = new MaterialMappingWindow(vm);
            window.Show();

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("MapeoMateriales failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }

    private static IReadOnlyList<string> GetProjectMaterials(Document doc)
        => new FilteredElementCollector(doc)
            .OfClass(typeof(Material))
            .Cast<Material>()
            .Select(m => m.Name)
            .Where(n => !string.IsNullOrWhiteSpace(n))
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
            .ToList();
}
