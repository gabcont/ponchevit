# 1. The Nature of the COVENIN Document (Domain Analysis)

The COVENIN normative document (specifically the E4 sector of Architectural Works) is not a simple flat catalog of items, but a polymorphic combinatorial matrix. It functions as a conditional decision tree where the rules of the game change depending on the context you find yourself in.

* **Variable Hierarchical Structure:** The columns are not static. If you select the activity "Construcción de paredes", the table requires you to define `Material`, `Acabado`, `Huecos`, and `Espesor`. But if you jump to "Herrería", the rules change and the norm requires defining `Elemento`, `Mecanismo`, and `Composición`.
* **Empty Bridges (Puentes Vacíos):** Physically in the PDF, there are cells that contain no information and do not contribute digits to the final code (e.g., columns of "vacant digits" or cells with nullity symbols), but they serve as a spatial bridge to connect a left attribute with a right one.
* **Cartesian Explosion:** If one were to attempt to flatten this document into a classic format (one row for each possible final item), we would generate a database with tens of thousands of highly redundant rows, making its maintenance and scalability impossible.

---

# 2. The Architectural Decision: Directed Acyclic Graph (DAG) & Flyweight Pattern

To digitize this complexity without breaking database normalization rules and to ensure optimal performance when integrating with BIM systems, the flat table model was discarded. Instead, the system was designed as a Directed Acyclic Graph (DAG) implemented on a relational engine (SQLite), utilizing the Flyweight Design Pattern.

This architecture strictly separates the Topology (the normative path) from the Semantics (the element data) to avoid Cross-Contamination (the risk of mistakenly assigning a floor finish to a wall `material` just because both are called "Mortero de cemento").

---

# 3. The Relational Schema (Three-Table Representation)

The graph materializes into three fundamental tables working in tandem:

### A. Columns Table (`Covenin_Columnas`)
It acts as the schema's metadata dictionary. 
It defines the "Types" or "Categories" of the variables being evaluated at any point in the tree. It stores universal names like `ACTIVIDAD`, `MATERIAL`, or `ESPESOR`. Its purpose is to dynamically feed the user interface (knowing what title to give the current dropdown menu) and to allow parameter crossing with the API.

### B. Values Table (`Covenin_Valores` - The Flyweight Store)
It is the single source of truth for semantic and parametric content. This is where the Flyweight pattern shines: an entity exists only once on the disk, regardless of how many times it appears in the norm.
* **Context Isolation:** It stores visual descriptions (e.g., "Bloques huecos de arcilla" or "15: e = 15 cm").
* **Parametric Data:** It is vital because it breaks down raw text into mathematical values (`Num_Min`, `Num_Max`, `Unidad`). This allows the parser to not only read text but also validate actual physical ranges (comparing if a geometry layer measures exactly 15.0 cm).

### C. Connections Table (`Covenin_Conexiones` - The Adjacency List)
It is the rule engine. It models the edges of the graph and strictly dictates the navigation direction.
* **Directional Control (`Parent_Id`):** Each record is a connection node pointing to its previous node. This guarantees that the option "Acabado Corriente" is only visible if the user comes from the parent node "Bloque de arcilla", respecting the matrix's path.
* **Code Contribution (`Codigo_Aportado`):** It stores the normative code fragment contributed by that particular decision (e.g., the digit 1 or 01).
* **Linkage:** It joins the structure with the content using foreign keys (`Id_Columna` and `Id_Valor_Asociado`).

---

# 4. How It Operates (Runtime Assembly)

When the system needs to build an item or populate a cascading menu, it executes a search through the connections. By traversing the `Parent_Id`, the algorithm concatenates the `Codigo_Aportado` of each visited edge from left to right.

If an "empty bridge" is detected during assembly, the connections table simply inherits the `Parent_Id` from its ancestor, skipping the void without losing the guiding thread. Finally, an algorithmic firewall stops any branch that reaches the 10 normative digits, preventing infinite recursions and validating that the final code (e.g., E411011015) is structurally perfect according to the original document.

# 5. Sample of the Three Tables in Action

tabla_columnas:
´´´
[
  {
    "Id_Columna": "COL_001",
    "Nombre": "CAPITULO"
  },
  {
    "Id_Columna": "COL_002",
    "Nombre": "SUB-CAPITULO"
  },
  {
    "Id_Columna": "COL_003",
    "Nombre": "ACTIVIDAD"
  },
  {
    "Id_Columna": "COL_004",
    "Nombre": "UN."
  },
  {
    "Id_Columna": "COL_005",
    "Nombre": "MATERIAL"
  },
  ...
´´´

tabla_conexiones:
´´´
[
  {
    "Id_Conexion": "CON_000001",
    "Parent_Id": null,
    "Codigo_Aportado": "E4",
    "Id_Columna": "COL_001",
    "Id_Valor_Asociado": null
  },
  {
    "Id_Conexion": "CON_000002",
    "Parent_Id": "CON_000001",
    "Codigo_Aportado": "1",
    "Id_Columna": "COL_002",
    "Id_Valor_Asociado": "VAL_00001"
  },
  {
    "Id_Conexion": "CON_000003",
    "Parent_Id": "CON_000002",
    "Codigo_Aportado": "1",
    "Id_Columna": "COL_003",
    "Id_Valor_Asociado": "VAL_00002"
  },
  {
    "Id_Conexion": "CON_000004",
    "Parent_Id": "CON_000003",
    "Codigo_Aportado": "",
    "Id_Columna": "COL_004",
    "Id_Valor_Asociado": "VAL_00003"
  },
  ´´´

tabla_valores:
´´´
[
  {
    "Id_Valor": "VAL_00001",
    "Descripcion_UI": "Albañilería",
    "Id_Columna": "COL_002",
    "Num_Min": null,
    "Num_Max": null,
    "Unidad": null
  },
  {
    "Id_Valor": "VAL_00002",
    "Descripcion_UI": "Construcción de paredes",
    "Id_Columna": "COL_003",
    "Num_Min": null,
    "Num_Max": null,
    "Unidad": null
  },
  {
    "Id_Valor": "VAL_00003",
    "Descripcion_UI": "m2",
    "Id_Columna": "COL_004",
    "Num_Min": null,
    "Num_Max": null,
    "Unidad": "m2"
  },
  {
    "Id_Valor": "VAL_00004",
    "Descripcion_UI": "Bloques huecos de arcilla",
    "Id_Columna": "COL_005",
    "Num_Min": null,
    "Num_Max": null,
    "Unidad": null
  },
  ...
´´´