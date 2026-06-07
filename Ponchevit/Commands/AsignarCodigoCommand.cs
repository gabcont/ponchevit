using System;
using System.Linq;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Ponchevit.Domain.Matching;
using Ponchevit.Revit;
using Ponchevit.Revit.Families;
using Ponchevit.Ui.PartidaSelection;

namespace Ponchevit.Commands;

[Transaction(TransactionMode.Manual)]
public class AsignarCodigoCommand : IExternalCommand
{
    public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
    {
        try
        {
            var services = App.Services;
            services.RevitContext.Attach(commandData.Application);

            var uidoc = commandData.Application.ActiveUIDocument;

            string? displayName = null;
            ElementTopology? topology = null;
            var selectedIds = uidoc.Selection.GetElementIds();
            ElementId? targetElementId = selectedIds.Count == 1 ? selectedIds.First() : null;

            if (targetElementId != null)
            {
                var el = uidoc.Document.GetElement(targetElementId);
                displayName = el?.Name;
                if (el != null)
                {
                    var topologyReader = new ElementTopologyReader(services.MaterialMappingResolver);
                    topology = topologyReader.Read(el);
                }
            }

            PartidaSelectionViewModel? vmRef = null;
            Action<AssignInput>? assignAction = null;

            if (targetElementId != null)
            {
                var capturedId = targetElementId;
                var capturedServices = services;
                assignAction = input =>
                    capturedServices.RevitContext.PostExternalEvent(doc =>
                    {
                        try
                        {
                            capturedServices.AssignCodeOrchestrator.Assign(doc, capturedId, input);
                            System.Windows.Application.Current?.Dispatcher.Invoke(() =>
                            {
                                if (vmRef == null) return;
                                vmRef.StatusIsError = false;
                                vmRef.StatusMessage = "Código asignado correctamente.";
                            });
                        }
                        catch (Exception ex)
                        {
                            capturedServices.Log.Error("Asignar Código: write failed.", ex);
                            var msg = ex.Message;
                            System.Windows.Application.Current?.Dispatcher.Invoke(() =>
                            {
                                if (vmRef == null) return;
                                vmRef.StatusIsError = true;
                                vmRef.StatusMessage = $"Error: {msg}";
                            });
                        }
                    });
            }

            var projectMaterials = services.ProjectMaterialQuery.GetProjectMaterials();

            bool canRecognize = topology != null && services.CanRecognizeTopology(topology);

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
                mode: WindowMode.Assign,
                targetElementDisplayName: displayName,
                assignAction: assignAction,
                targetTopology: topology,
                recognizeFunc: canRecognize ? services.RecognizeTopology : null);

            vmRef = vm;

            var window = new PartidaSelectionWindow(
                vm,
                services.CoveninRulesRepository,
                services.MaterialMappingRepository,
                services.RevitContext,
                services.ProjectMaterialQuery,
                services.Log);

            window.Show();

            return Result.Succeeded;
        }
        catch (Exception ex)
        {
            App.Services?.Log.Error("AsignarCodigo failed.", ex);
            message = ex.Message;
            return Result.Failed;
        }
    }
}
