# Plugin UI Specification and Business Logic

This document details the graphical user interface (GUI) specification and the underlying business logic for the three main workflows of the plugin. Access to these modules is managed via a native Ribbon (options bar) in Autodesk Revit, which exposes three main commands: `Agregar familia`, `Asignar Código`, and `Reconocer elemento`.

---

## 1. Use Case 1: Parametric Generation (`Agregar Familia`)
This module acts as a factory (`Factory`) controlled by the norm. Its goal is to instantiate new elements into the Revit model, ensuring they are born with the correct geometry and metadata according to COVENIN rules.

### 1.1. Interface Structure (Three-Panel View)
The main window is split into three sequential vertical sections:
* **Left Panel (Hierarchical Navigation):** Presents a navigation tree based on the database structure: `Capítulo` > `Subcapítulo` > `Sección`. Only sections for which the plugin has implemented programmatic generation routines will be enabled (e.g., `Muros`, `Puertas`, `Ventanas`).
* **Central Panel (Parameter Resolution):** Activates upon selecting a terminal section in the left panel. The system traverses the Directed Acyclic Graph (DAG) and dynamically generates controls (dropdown menus) for the remaining required parameters along that path (e.g., `Mecanismo`, `Composición`, `Materiales`).
  * *Special Controls:* Includes a "`Subir modelo 3D`" button allowing the user to inject a `.rfa` file (`Familia Base`) or load external resources when the element to be generated requires a complex 3D topology that cannot be generated entirely via code (e.g., sinks, swing doors).
* **Right Panel (Results Table):** Displays an interactive grid with the resulting `partidas` that match the selected parameters. It allows individual or mass selection using checkboxes.

### 1.2. Execution Logic
When clicking "`Agregar`", the plugin reads the selected elements. For system families (such as `Muros`), it assembles the internal structure (`CompoundStructure`) using the physical data from the database. For loadable families, it duplicates the "`Modelo 3D`" provided by the user and adjusts its dimensional parameters while injecting the COVENIN metadata into the newly generated instance.

---

## 2. Use Case 2: Guided Manual Enrichment (`Asignar Código`)
This module reuses the visual architecture of Case 1 but radically changes its internal behavior. Its objective is not to create new geometry, but to catalog elements that the architect has already previously modeled in the project.

### 2.1. Interface Differences
* **Full Enablement:** Unlike Case 1, the left navigation panel has all sections of the norm enabled. Since the plugin does not need to know how to model the element, it can assign codes from any chapter (e.g., waterproofing, paints, minor ironmongery).
* **Target Selector:** In the lower section of the central panel, the "`Elemento a codificar`" control is incorporated. This component registers the user's current selection on the Revit canvas.

### 2.2. Execution Logic
When clicking "`Asignar`", the system extracts the resulting COVENIN code string from the right panel's matrix and injects it directly into the `Shared Parameters` (Compartidos) of the previously selected element in the model. This democratizes coding, enabling the tagging of complex `partidas` without requiring geometric generation algorithms.

---

## 3. Use Case 3: Audit and Intelligent Mapping (`Reconocer Elemento`)
This is the intelligent assistance module. It inverts the traditional workflow: instead of the user searching the norm to apply it to the model, the model queries the norm to validate itself.

### 3.1. Interface Structure (Comparative Scanner)
It presents a compact modal window ("`Reconociendo elemento...`") divided into two blocks of comparative information:
* **"`Tu Elemento`" Panel:** Displays a summary of the physical and categorical properties read directly from the Revit API of the selected element (e.g., Category: `Puerta`, Width: 0.90m, Material: `Madera`).
* **"`Partidas COVENIN coincidentes`" Panel:** Displays a filtered list of normative items.

### 3.2. Execution Logic (The Detection Algorithm)
1. The user selects a physical element in the model and presses the button on the Ribbon.
2. The plugin reads the element's topology (category, layers, materials, dimensions).
3. The rules engine queries the unified SQLite database and filters out all branches of the graph that are incompatible with the element's physics.
4. The system returns the viable options to the right panel.
5. The user makes the final decision by selecting the correct `partida` and presses "`Asignar código`", injecting the metadata into the element.
