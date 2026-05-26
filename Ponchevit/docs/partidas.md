# Hierarchical Structure of the `Partidas` List

To efficiently manage the complete catalog of the norm, the unified database relies on a hierarchical coding system that classifies each `partida` into three logical levels. Each partida by itself does not have this information but the masks of each capitulo, subcapitulo and seccion will be stored for its use as a mask for querying.

It is important to highlight that the digit length at each level is not static or fixed. The COVENIN norm expands its codes according to the depth of the specialty, so the system is designed to read the hierarchy from left to right without depending on a strict number of characters.

The structure is divided as follows:

### 1. `Capítulo` (Root Level)
Defines the macro-category of the work. It is the starting point of the entire tree.
* **Example:** E4 (Obras Arquitectónicas).

### 2. `Subcapítulo` (Specialty Level)
Groups the activities by construction disciplines or trades within the main `Capítulo`.
* **Examples:** E41 (Albañilería), E43 (Herrería).

### 3. `Sección` (Specific Activity Level)
Details the exact work front or the construction element to be quantified. This level can have a variable length of numeric digits depending on how specific the `partida` is in the original norm.
* **Examples:** E411 (Construcción de paredes), E43701 (Puertas metálicas).

---

# Technical Benefits of this Structure

Modeling the data under this three-part categorization offers direct advantages for the plugin's performance:

* **Efficient Cascading Menus:** The graphical interface does not need to load thousands of `partidas` all at once. By logically segmenting from the root, the dropdown menus are populated progressively. The user first filters the `Capítulo`, then the `Subcapítulo`, and finally the `Sección`, optimizing memory and improving the user experience.
* **Prefix Masking Search:** Because the codes grow sequentially, the system can apply Prefix Masking to perform ultra-fast searches. If the user selects or types the root E437, the database can instantly isolate all the `partidas`, routes, and graph nodes that begin exactly with that sequence, discarding the rest of the document in milliseconds.