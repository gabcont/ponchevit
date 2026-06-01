using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.ExtensibleStorage;
using Ponchevit.Data;

namespace Ponchevit.Revit.Materials;

/// <summary>
/// Backs IMaterialMappingRepository using Revit ExtensibleStorage (Schema + one DataStorage
/// element per document). The mapping travels with the .rvt over cloud workshare.
///
/// Schema GUID is PERMANENT — changing it creates a new orphan schema; old mappings
/// become invisible. See ADR 2026-05-31 — GUID source-of-truth.
/// </summary>
public sealed class ExtensibleStorageMaterialMappingRepository : IMaterialMappingRepository
{
    // ── Stable GUID — NEVER CHANGE ──────────────────────────────────────────────
    public static readonly Guid SchemaGuid = new("E5F6A7B8-C9D0-1234-EF01-234567890123");
    // ────────────────────────────────────────────────────────────────────────────

    private const string SchemaName = "PonchevitMaterialMapping";
    private const string FieldName  = "Entries";

    private readonly Func<Document> _getDocument;

    public ExtensibleStorageMaterialMappingRepository(Func<Document> getDocument)
        => _getDocument = getDocument;

    public IReadOnlyDictionary<string, string> GetAll()
    {
        var doc = _getDocument();
        Schema? schema = Schema.Lookup(SchemaGuid);
        if (schema == null)
            return new Dictionary<string, string>();

        DataStorage? storage = FindStorage(doc, schema);
        if (storage == null)
            return new Dictionary<string, string>();

        Entity entity = storage.GetEntity(schema);
        if (!entity.IsValid())
            return new Dictionary<string, string>();

        IDictionary<string, string> map =
            entity.Get<IDictionary<string, string>>(FieldName)
            ?? new Dictionary<string, string>();

        return new Dictionary<string, string>(map);
    }

    public void Set(string revitMaterialName, string coveninValueId)
    {
        var dict = new Dictionary<string, string>(GetAll())
        {
            [revitMaterialName] = coveninValueId
        };
        Save(dict);
    }

    public void Remove(string revitMaterialName)
    {
        var dict = new Dictionary<string, string>(GetAll());
        dict.Remove(revitMaterialName);
        Save(dict);
    }

    public void Clear() => Save(new Dictionary<string, string>());

    private void Save(Dictionary<string, string> dict)
    {
        Document doc = _getDocument();
        Schema schema = GetOrCreateSchema();
        DataStorage? storage = FindStorage(doc, schema);

        using var t = new Transaction(doc, "Save Ponchevit material mapping");
        t.Start();

        if (storage == null)
            storage = DataStorage.Create(doc);

        var entity = new Entity(schema);
        entity.Set<IDictionary<string, string>>(FieldName, dict);
        storage.SetEntity(entity);

        t.Commit();
    }

    private static DataStorage? FindStorage(Document doc, Schema schema)
        => new FilteredElementCollector(doc)
            .OfClass(typeof(DataStorage))
            .Cast<DataStorage>()
            .FirstOrDefault(ds => ds.GetEntity(schema).IsValid());

    private static Schema GetOrCreateSchema()
    {
        Schema? existing = Schema.Lookup(SchemaGuid);
        if (existing != null)
            return existing;

        var sb = new SchemaBuilder(SchemaGuid);
        sb.SetSchemaName(SchemaName);
        sb.SetReadAccessLevel(AccessLevel.Public);
        sb.SetWriteAccessLevel(AccessLevel.Public);
        sb.AddMapField(FieldName, typeof(string), typeof(string));
        return sb.Finish();
    }
}
