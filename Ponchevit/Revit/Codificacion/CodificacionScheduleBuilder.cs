using System;
using System.Linq;
using Autodesk.Revit.DB;
using Ponchevit.Infrastructure;
using Ponchevit.Revit.SharedParameters;

namespace Ponchevit.Revit.Codificacion;

/// <summary>
/// Creates a multi-category ViewSchedule listing the 4 COVENIN shared parameters
/// alongside family/type name. Returns the schedule name on success; throws on failure
/// so callers can surface errors to the user.
/// </summary>
public sealed class CodificacionScheduleBuilder
{
    private readonly ILog _log;

    public CodificacionScheduleBuilder(ILog log)
        => _log = log ?? throw new ArgumentNullException(nameof(log));

    public string Build(Document doc)
    {
        if (doc == null) throw new ArgumentNullException(nameof(doc));

        CoveninParameters.EnsureBoundToProject(doc);

        using var t = new Transaction(doc, "Generar Schedule COVENIN");
        t.Start();

        var schedule = ViewSchedule.CreateSchedule(doc, ElementId.InvalidElementId);
        string scheduleName = "COVENIN - Codificación " + DateTime.Now.ToString("yyyy-MM-dd HH-mm");
        schedule.Name = scheduleName;

        var schedulableFields = schedule.Definition.GetSchedulableFields();

        // Add the 4 COVENIN shared params by matching their stable GUIDs.
        // Family/Type name columns are intentionally excluded: including them would
        // produce one row per Revit type rather than one row per COVENIN code.
        ScheduleField? codeScheduleField = null;
        foreach (var guid in new[]
        {
            CoveninParameters.CodigoCompletoGuid,
            CoveninParameters.CapituloGuid,
            CoveninParameters.SubcapituloGuid,
            CoveninParameters.SeccionGuid,
        })
        {
            var sf = schedulableFields.FirstOrDefault(f =>
                doc.GetElement(f.ParameterId) is SharedParameterElement spe
                && spe.GuidValue == guid);

            if (sf != null)
            {
                var added = schedule.Definition.AddField(sf);
                if (guid == CoveninParameters.CodigoCompletoGuid)
                    codeScheduleField = added;
            }
        }

        // Count field — graceful skip if not found.
        var countField = schedulableFields.FirstOrDefault(
            f => f.FieldType == ScheduleFieldType.Count);
        if (countField != null)
            schedule.Definition.AddField(countField);

        // One row per unique COVENIN code: group by code, collapse instances.
        if (codeScheduleField != null)
        {
            var sgf = new ScheduleSortGroupField(codeScheduleField.FieldId, ScheduleSortOrder.Ascending);
            schedule.Definition.AddSortGroupField(sgf);
        }
        schedule.Definition.IsItemized = false;

        t.Commit();
        _log.Info($"CodificacionScheduleBuilder: created schedule '{scheduleName}'.");
        return scheduleName;
    }
}
