using Autodesk.Revit.UI;
using System;

namespace Ponchevit.Core
{
    public class CoveninExternalEventHandler : IExternalEventHandler
    {
        private Action<UIApplication> _pendingAction;
        private readonly object _lock = new object();

        public void SetAction(Action<UIApplication> action)
        {
            lock (_lock) { _pendingAction = action; }
        }

        public void Execute(UIApplication app)
        {
            Action<UIApplication> actionToExecute;
            lock (_lock)
            {
                actionToExecute = _pendingAction;
                _pendingAction = null;
            }

            if (actionToExecute == null) return;

            try { actionToExecute(app); }
            catch (Exception ex) { TaskDialog.Show("Error", ex.Message); }
        }

        public string GetName() => "CoveninExternalEventHandler";
    }
}