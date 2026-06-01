using System;
using System.Collections.Generic;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace Ponchevit.Revit.Context;

/// <summary>
/// Thin wrapper over the Revit application context. Commands call Attach() on startup
/// so that modeless windows can post work back to Revit's main thread without holding
/// a direct RevitAPI reference.
/// </summary>
public interface IRevitContext
{
    UIDocument ActiveUiDocument { get; }
    IReadOnlyList<ElementId> GetSelectedElementIds();

    /// <summary>
    /// Queues <paramref name="work"/> to run on Revit's main thread via ExternalEvent.
    /// Use this from modeless WPF windows to write to the document.
    /// </summary>
    void PostExternalEvent(Action<Document> work);
}
