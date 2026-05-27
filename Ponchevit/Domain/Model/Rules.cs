namespace Ponchevit.Domain.Model;

/// <summary>
/// Defines a variable category being evaluated in the COVENIN DAG (e.g., MATERIAL, ESPESOR).
/// </summary>
public record Columna(string IdColumna, string Nombre);

/// <summary>
/// A semantic value associated with a column, potentially containing parametric data.
/// </summary>
public record Valor(
    string IdValor,
    string DescripcionUi,
    string IdColumna,
    double? NumMin = null,
    double? NumMax = null,
    string? Unidad = null
);

/// <summary>
/// An edge in the DAG connecting a parent state to a column/value choice.
/// </summary>
public record Conexion(
    string IdConexion,
    string? ParentId,
    string CodigoAportado,
    string IdColumna,
    string? IdValorAsociado
);
