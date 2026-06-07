namespace Ponchevit.Domain.Codificacion;

public sealed record CodificacionSummary(
    string FamilyTypeName,
    string CategoryDisplayName,
    string? CodigoCompleto,
    int InstanceCount,
    double QuantityValue,
    string QuantityUnit,
    bool IsCodified,
    long SampleInstanceId);
