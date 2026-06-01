using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace Ponchevit.Revit.Context;

/// <summary>
/// Concrete IRevitContext. Created once in Services.Build() so the ExternalEvent is
/// registered at add-in startup. Commands call Attach(uiApp) before showing windows.
/// </summary>
public sealed class RevitContextImpl : IRevitContext
{
    private readonly RevitEventHandler _handler;
    private readonly ExternalEvent _externalEvent;
    private UIApplication? _app;

    public RevitContextImpl()
    {
        _handler = new RevitEventHandler();
        _externalEvent = ExternalEvent.Create(_handler);
    }

    /// <summary>Called by each IExternalCommand before creating windows.</summary>
    public void Attach(UIApplication app) => _app = app;

    public UIDocument ActiveUiDocument
        => _app?.ActiveUIDocument
           ?? throw new InvalidOperationException(
               "No active Revit session. RevitContextImpl.Attach() must be called from an IExternalCommand first.");

    public IReadOnlyList<ElementId> GetSelectedElementIds()
        => ActiveUiDocument.Selection.GetElementIds().ToList();

    public void PostExternalEvent(Action<Document> work)
    {
        _handler.Enqueue(work);
        _externalEvent.Raise();
    }
}

internal sealed class RevitEventHandler : IExternalEventHandler
{
    private Action<Document>? _pending;

    internal void Enqueue(Action<Document> work)
        => Interlocked.Exchange(ref _pending, work);

    public void Execute(UIApplication app)
    {
        var work = Interlocked.Exchange(ref _pending, null);
        work?.Invoke(app.ActiveUIDocument.Document);
    }

    public string GetName() => "Ponchevit";
}
