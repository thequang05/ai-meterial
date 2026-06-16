LOAD CSV WITH HEADERS FROM 'file:///materials.csv' AS row
CALL {
    WITH row

    MERGE (m:Material {uid: row.uid})
    SET
        m.source_id = row.source_id,
        m.formula =
            CASE
                WHEN row.formula IS NULL OR trim(row.formula) = "" THEN null
                ELSE trim(row.formula)
            END,
        m.formation_energy_per_atom =
            CASE
                WHEN row.formation_energy_per_atom IS NULL OR trim(row.formation_energy_per_atom) = "" THEN null
                ELSE toFloat(row.formation_energy_per_atom)
            END,
        m.num_atoms =
            CASE
                WHEN row.num_atoms IS NULL OR trim(row.num_atoms) = "" THEN null
                ELSE toInteger(row.num_atoms)
            END,
        m.num_edges =
            CASE
                WHEN row.num_edges IS NULL OR trim(row.num_edges) = "" THEN null
                ELSE toInteger(row.num_edges)
            END,
        m.graph_index =
            CASE
                WHEN row.graph_index IS NULL OR trim(row.graph_index) = "" THEN null
                ELSE toInteger(row.graph_index)
            END,
        m.cif_parse_warning =
            CASE
                WHEN row.cif_parse_warning IS NULL OR trim(row.cif_parse_warning) = "" THEN null
                ELSE row.cif_parse_warning
            END,
        m.cif_parse_error =
            CASE
                WHEN row.cif_parse_error IS NULL OR trim(row.cif_parse_error) = "" THEN null
                ELSE row.cif_parse_error
            END,
        m.has_cif_warning =
            CASE
                WHEN row.cif_parse_warning IS NULL OR trim(row.cif_parse_warning) = "" THEN false
                ELSE true
            END

    MERGE (src:Source {name: row.source})
    MERGE (m)-[:FROM_SOURCE]->(src)

    FOREACH (_ IN CASE
        WHEN row.formula IS NOT NULL AND trim(row.formula) <> "" THEN [1]
        ELSE []
    END |
        MERGE (f:Formula {value: trim(row.formula)})
        MERGE (m)-[:HAS_FORMULA]->(f)
    )

    MERGE (st:Structure {structure_id: row.uid + "_structure"})
    SET
        st.lattice_a =
            CASE
                WHEN row.lattice_a IS NULL OR trim(row.lattice_a) = "" THEN null
                ELSE toFloat(row.lattice_a)
            END,
        st.lattice_b =
            CASE
                WHEN row.lattice_b IS NULL OR trim(row.lattice_b) = "" THEN null
                ELSE toFloat(row.lattice_b)
            END,
        st.lattice_c =
            CASE
                WHEN row.lattice_c IS NULL OR trim(row.lattice_c) = "" THEN null
                ELSE toFloat(row.lattice_c)
            END,
        st.alpha =
            CASE
                WHEN row.alpha IS NULL OR trim(row.alpha) = "" THEN null
                ELSE toFloat(row.alpha)
            END,
        st.beta =
            CASE
                WHEN row.beta IS NULL OR trim(row.beta) = "" THEN null
                ELSE toFloat(row.beta)
            END,
        st.gamma =
            CASE
                WHEN row.gamma IS NULL OR trim(row.gamma) = "" THEN null
                ELSE toFloat(row.gamma)
            END,
        st.volume =
            CASE
                WHEN row.volume IS NULL OR trim(row.volume) = "" THEN null
                ELSE toFloat(row.volume)
            END,
        st.density =
            CASE
                WHEN row.density IS NULL OR trim(row.density) = "" THEN null
                ELSE toFloat(row.density)
            END,
        st.num_atoms =
            CASE
                WHEN row.num_atoms IS NULL OR trim(row.num_atoms) = "" THEN null
                ELSE toInteger(row.num_atoms)
            END,
        st.num_edges =
            CASE
                WHEN row.num_edges IS NULL OR trim(row.num_edges) = "" THEN null
                ELSE toInteger(row.num_edges)
            END

    MERGE (m)-[:HAS_STRUCTURE]->(st)

    MERGE (p:PropertyMeasurement {
        property_id: row.uid + "_formation_energy_per_atom"
    })
    SET
        p.name = "formation_energy_per_atom",
        p.value =
            CASE
                WHEN row.formation_energy_per_atom IS NULL OR trim(row.formation_energy_per_atom) = "" THEN null
                ELSE toFloat(row.formation_energy_per_atom)
            END,
        p.unit = "eV/atom",
        p.source = row.source

    MERGE (m)-[:HAS_PROPERTY]->(p)

    WITH row, m
    UNWIND split(row.elements_str, ";") AS symbol
    WITH m, trim(symbol) AS symbol
    WHERE symbol <> ""

    MERGE (e:Element {symbol: symbol})
    MERGE (m)-[:CONTAINS_ELEMENT]->(e)
}
IN TRANSACTIONS OF 2000 ROWS;