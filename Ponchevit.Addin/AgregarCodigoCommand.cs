using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Ponchevit.Presentation.Views;
using Microsoft.Extensions.DependencyInjection;


namespace Ponchevit.Addin.Commands
{
    [Autodesk.Revit.Attributes.Transaction(Autodesk.Revit.Attributes.TransactionMode.Manual)]
    public class AgregarFamiliaCommand : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            TaskDialog.Show("Ponchevit", "Ejecutando: Agregar Familia");
            return Result.Succeeded;
        }
    }
}

namespace Ponchevit.Addin.Commands
{
    [Autodesk.Revit.Attributes.Transaction(Autodesk.Revit.Attributes.TransactionMode.Manual)]
    public class AsignarCodigoCommand : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            TaskDialog.Show("Ponchevit", "Ejecutando: Asignar Código");
            return Result.Succeeded;
        }
    }
}

namespace Ponchevit.Addin.Commands
{
    [Autodesk.Revit.Attributes.Transaction(Autodesk.Revit.Attributes.TransactionMode.Manual)]
    public class GenerarAutomaticoCommand : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            TaskDialog.Show("Ponchevit", "Ejecutando: Generar Automático");
            return Result.Succeeded;
        }
    }
}