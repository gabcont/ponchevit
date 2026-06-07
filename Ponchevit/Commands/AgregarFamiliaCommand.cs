using System;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Ponchevit.Ui.PartidaSelection;

namespace Ponchevit.Commands;

[Transaction(TransactionMode.Manual)]
public class AgregarFamiliaCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);

            // Fix #6 / Fix E: IProjectMaterialQuery obtains Document internally via
            // IRevitContext — the command no longer needs to extract a Document to pass through.
            var projectMaterials = services.ProjectMaterialQuery.GetProjectMaterials();

            var vm = new PartidaSelectionViewModel(
                services.PartidasRepository,
                services.CoveninRulesRepository,
                services.PartidaCatalog,
                services.ConstructibilityResolver,
                services.MaterialMappingResolver,
                services.MaterialMappingRepository,
                services.RevitContext,
                services.FamilyGenerators,
                services.GenerationOrchestrator,
                projectMaterials,
                services.HierarchyResolver,
                services.Log,
                mode: WindowMode.Generate,
                targetElementDisplayName: null,
                assignAction: null);

            var window = new PartidaSelectionWindow(
                vm,
                services.CoveninRulesRepository,
                services.MaterialMappingRepository,
                services.RevitContext,
                services.ProjectMaterialQuery,
                services.Log);

            window.Show(); // Modeless — returns immediately.

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("AgregarFamilia failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }
}
