using Ponchevit.Domain.Codificacion;

namespace Ponchevit.Ui.Codificacion;

public sealed class CodificacionRowViewModel
{
    public string FamilyTypeName     { get; }
    public string CategoryDisplayName { get; }
    public string CodigoDisplay      { get; }
    public string QuantityDisplay    { get; }
    public bool   IsCodified         { get; }
    public long   SampleInstanceId   { get; }

    internal CodificacionSummary Source { get; }

    public CodificacionRowViewModel(CodificacionSummary summary)
    {
        Source              = summary;
        FamilyTypeName      = summary.FamilyTypeName;
        CategoryDisplayName = summary.CategoryDisplayName;
        CodigoDisplay       = summary.CodigoCompleto ?? "Sin código";
        QuantityDisplay     = $"{summary.QuantityValue:F1} {summary.QuantityUnit}";
        IsCodified          = summary.IsCodified;
        SampleInstanceId    = summary.SampleInstanceId;
    }
}
