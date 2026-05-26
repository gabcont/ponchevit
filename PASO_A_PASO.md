# Paso a paso desde cero

## 1. Instalar lo necesario
- Revit 2026 instalado localmente.
- .NET 8 SDK.
- Visual Studio o VS Code con extensión de C#.

## 2. Crear la solución y proyectos
- Crear la solución `Ponchevit.sln`.
- Crear estos proyectos:
  - `Ponchevit.Core`
  - `Ponchevit.Data`
  - `Ponchevit.Presentation`
  - `Ponchevit.Addin`
- Agregar los proyectos a la solución.

## 3. Configurar `Ponchevit.Core`
- Abrir `Ponchevit.Core\Ponchevit.Core.csproj`.
- Usar `net8.0-windows`.
- Activar `ImplicitUsings` y `Nullable`.
- Agregar `Nice3point.Revit.Sdk`.
- Agregar referencias a:
  - `C:\Program Files\Autodesk\Revit 2026\RevitAPI.dll`
  - `C:\Program Files\Autodesk\Revit 2026\RevitAPIUI.dll`

## 4. Crear el handler de eventos
- Crear `Ponchevit.Core\CoveninExternalEventHandler.cs`.
- Implementar `IExternalEventHandler`.
- Usar `UIApplication` como parámetro.
- Guardar una `Action<UIApplication>` pendiente.
- Ejecutarla dentro de `Execute`.

## 5. Configurar `Ponchevit.Presentation`
- Abrir `Ponchevit.Presentation\Ponchevit.Presentation.csproj`.
- Usar `net8.0-windows`.
- Activar `UseWPF`.
- Crear `Ponchevit.Presentation\Views\AgregarCodigoView.cs`.
- Hacer que sea una `Window` simple.

## 6. Configurar `Ponchevit.Addin`
- Abrir `Ponchevit.Addin\Ponchevit.Addin.csproj`.
- Usar `net8.0-windows`.
- Agregar referencias a:
  - `Ponchevit.Core`
  - `Ponchevit.Data`
  - `Ponchevit.Presentation`
- Agregar `Nice3point.Revit.Sdk`.
- Agregar `Microsoft.Extensions.DependencyInjection`.
- Agregar referencias a:
  - `C:\Program Files\Autodesk\Revit 2026\RevitAPI.dll`
  - `C:\Program Files\Autodesk\Revit 2026\RevitAPIUI.dll`
- Marcar `Resources\*.xaml` como `EmbeddedResource`.

## 7. Crear `App.cs`
- Implementar `IExternalApplication`.
- En `OnStartup`:
  - crear `ServiceCollection`
  - crear `CoveninExternalEventHandler`
  - crear `ExternalEvent`
  - registrar el handler
  - registrar `AgregarCodigoView`
  - construir `ServiceProvider`
  - llamar a `RegisterRibbon`

## 8. Registrar el ribbon
- En `App.cs`, crear `RegisterRibbon(UIControlledApplication application)`.
- Crear la pestaña `Ponchevit`.
- Crear el panel `Codificación`.
- Crear 3 botones con `PushButtonData`:
  - `Agregar\nFamilia`
  - `Asignar\nCódigo`
  - `Generar\nAutomático`
- Enlazar cada botón con su clase en `Ponchevit.Addin.Commands`.
- Agregar cada botón con `panel.AddItem(...)`.

## 9. Crear los comandos
- Crear el espacio de nombres `Ponchevit.Addin.Commands`.
- Crear estas clases con `IExternalCommand`:
  - `AgregarFamiliaCommand`
  - `AsignarCodigoCommand`
  - `GenerarAutomaticoCommand`
- En esta versión inicial, cada comando puede mostrar un `TaskDialog`.

## 10. Agregar iconos incrustados
- Crear `Ponchevit.Addin\Resources\Add.xaml`.
- Crear `Ponchevit.Addin\Resources\Code.xaml`.
- Crear `Ponchevit.Addin\Resources\Lightning.xaml`.
- Definir cada icono como `DrawingImage`.
- No usar PNG sueltos si se quiere que queden embebidos.

## 11. Cargar imágenes desde recursos
- En `App.cs`, crear `GetImage(string resourceName)`.
- Leer el recurso con `Assembly.GetExecutingAssembly().GetManifestResourceStream(...)`.
- Cargar el XAML con `XamlReader.Load(stream)`.
- Asignar el resultado a `LargeImage` e `Image`.

## 12. Puntos críticos para que funcione
- El nombre del recurso debe coincidir con el namespace y la ruta real.
- Si el icono no aparece, revisar `EmbeddedResource` y el nombre exacto del archivo.
- Si no compila, revisar que `RevitAPI.dll` y `RevitAPIUI.dll` existan en la ruta local.
- Si un botón falla al ejecutar, revisar que exista la clase en `Ponchevit.Addin.Commands` con el nombre exacto usado en `PushButtonData`.

## 13. Validar
- Ejecutar:
  - `dotnet build Ponchevit.Core\Ponchevit.Core.csproj -clp:ErrorsOnly`
  - `dotnet build Ponchevit.Presentation\Ponchevit.Presentation.csproj -clp:ErrorsOnly`
  - `dotnet build Ponchevit.Addin\Ponchevit.Addin.csproj -clp:ErrorsOnly`
- Si todo está bien, el build termina sin errores.