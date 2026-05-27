namespace Ponchevit.Domain.Model;

/// <summary>
/// Represents the root level of the COVENIN hierarchy (e.g., E4).
/// </summary>
public record Capitulo(string Id, string Codigo, string Titulo);

/// <summary>
/// Represents the specialty level within a Capítulo (e.g., E41).
/// </summary>
public record Subcapitulo(string Id, string CapituloId, string Codigo, string Titulo);

/// <summary>
/// Represents the specific activity level (e.g., E411).
/// </summary>
public record Seccion(string Id, string CapituloId, string SubcapituloId, string Codigo, string Titulo);

/// <summary>
/// Represents a final work item (Partida) with its associated metadata.
/// </summary>
public record Partida(
    string CodigoPartida,
    string Unidad,
    string Descripcion,
    string Capitulo,
    string? Subcapitulo = null,
    string? Seccion = null
);
