using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Ponchevit.Revit.Context;

namespace Ponchevit.Revit;

/// <summary>
/// Concrete implementation of <see cref="IProjectMaterialQuery"/> backed by
/// <see cref="FilteredElementCollector"/>.  Lives in Revit/ — the only layer
/// that may reference RevitAPI directly.
/// Fix E: Document is obtained via IRevitContext, eliminating the Document parameter
/// from the public API and preventing callers from needing to know about Revit Document.
/// </summary>
public sealed class ProjectMaterialQuery : IProjectMaterialQuery
{
    private readonly IRevitContext _revitContext;

    public ProjectMaterialQuery(IRevitContext revitContext)
    {
        _revitContext = revitContext ?? throw new ArgumentNullException(nameof(revitContext));
    }

    public IReadOnlyList<string> GetProjectMaterials()
    {
        var doc = _revitContext.ActiveUiDocument?.Document;
        if (doc == null)
            return Array.Empty<string>();

        return new FilteredElementCollector(doc)
            .OfClass(typeof(Material))
            .Cast<Material>()
            .Select(m => m.Name)
            .Where(n => !string.IsNullOrWhiteSpace(n))
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}
