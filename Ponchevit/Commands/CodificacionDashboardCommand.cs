using System;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Ponchevit.Ui.Codificacion;

namespace Ponchevit.Commands;

[Transaction(TransactionMode.ReadOnly)]
public class CodificacionDashboardCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);

            var doc = commandData.Application.ActiveUIDocument.Document;

            var summaries       = services.ProjectInventoryReader.Read(doc);
            var projectMaterials = services.ProjectMaterialQuery.GetProjectMaterials();

            var vm = new CodificacionDashboardViewModel(
                summaries,
                projectMaterials,
                services.PartidasRepository,
                services.CoveninRulesRepository,
                services.PartidaCatalog,
                services.ConstructibilityResolver,
                services.MaterialMappingResolver,
                services.MaterialMappingRepository,
                services.RevitContext,
                services.FamilyGenerators,
                services.GenerationOrchestrator,
                services.HierarchyResolver,
                services.AssignCodeOrchestrator,
                services.ProjectMaterialQuery,
                services.ProjectInventoryReader,
                services.CodificacionScheduleBuilder,
                services.RecognizeTopology,
                services.CanRecognizeTopology,
                services.Log);

            var window = new CodificacionDashboardWindow(vm);
            window.Show();

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("CodificacionDashboard failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }
}
