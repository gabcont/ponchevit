using System;

namespace Ponchevit.Domain.Model;

/// <summary>
/// Strongly-typed wrapper for a COVENIN code string.
/// </summary>
public readonly record struct CodigoCovenin : IComparable<CodigoCovenin>, IComparable
{
    public string Value { get; init; }

    public CodigoCovenin(string value)
    {
        Value = value ?? string.Empty;
    }

    public int Length => Value.Length;

    public bool IsEmpty => string.IsNullOrWhiteSpace(Value);

    public override string ToString() => Value;

    public int CompareTo(CodigoCovenin other)
    {
        return string.Compare(Value, other.Value, StringComparison.OrdinalIgnoreCase);
    }

    public int CompareTo(object? obj)
    {
        if (obj is null) return 1;
        if (obj is not CodigoCovenin other) throw new ArgumentException("Object is not a CodigoCovenin");
        return CompareTo(other);
    }

    public static implicit operator string(CodigoCovenin code) => code.Value;
    public static implicit operator CodigoCovenin(string value) => new(value);

    // Basic prefix helpers as suggested by roadmap 1.1 description
    public bool StartsWith(string prefix) => Value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
}
