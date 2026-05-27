using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Autodesk.Revit.Attributes;

namespace Ponchevit.Commands
{
    [Transaction(TransactionMode.ReadOnly)]
    public class CommandOne : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            TaskDialog.Show("Command One", "Command One executed successfully.");
            return Result.Succeeded;
        }
    }
}