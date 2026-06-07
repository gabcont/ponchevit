using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace Ponchevit.Ui.Codificacion;

/// <summary>
/// Converts a null or empty string to Visibility.Collapsed; any non-empty string → Visible.
/// </summary>
[ValueConversion(typeof(string), typeof(Visibility))]
public sealed class NullOrEmptyToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => string.IsNullOrEmpty(value as string) ? Visibility.Collapsed : Visibility.Visible;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

/// <summary>
/// Converts between <see cref="StatusFilter"/> and a RadioButton's IsChecked state.
/// ConverterParameter is the string name of the <see cref="StatusFilter"/> value the
/// RadioButton represents.
/// </summary>
[ValueConversion(typeof(StatusFilter), typeof(bool))]
public sealed class StatusFilterToBoolConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is StatusFilter filter && parameter is string paramStr
            && Enum.TryParse<StatusFilter>(paramStr, out var target))
        {
            return filter == target;
        }
        return false;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is bool isChecked && isChecked
            && parameter is string paramStr
            && Enum.TryParse<StatusFilter>(paramStr, out var target))
        {
            return target;
        }
        return Binding.DoNothing;
    }
}
