using Ponchevit.Domain.Model;

namespace Ponchevit.Revit.Families;

public sealed record AssignInput(
    CodigoCovenin Codigo,
    string Capitulo,
    string Subcapitulo,
    string Seccion,
    string Descripcion);
