CREATE CONSTRAINT material_uid IF NOT EXISTS
FOR (m:Material)
REQUIRE m.uid IS UNIQUE;

CREATE CONSTRAINT element_symbol IF NOT EXISTS
FOR (e:Element)
REQUIRE e.symbol IS UNIQUE;

CREATE CONSTRAINT source_name IF NOT EXISTS
FOR (s:Source)
REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT formula_value IF NOT EXISTS
FOR (f:Formula)
REQUIRE f.value IS UNIQUE;

CREATE CONSTRAINT structure_id IF NOT EXISTS
FOR (st:Structure)
REQUIRE st.structure_id IS UNIQUE;

CREATE CONSTRAINT property_id IF NOT EXISTS
FOR (p:PropertyMeasurement)
REQUIRE p.property_id IS UNIQUE;