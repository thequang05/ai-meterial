"""
Neo4j retrieval layer for materials inverse design.

This module wraps the Neo4j knowledge graph (built in phase_1) and exposes
read-only query helpers focused on *formation energy* — the central property
for inverse design (lower / more negative formation energy => more
thermodynamically stable).

Graph schema (from phase_1/import.cypher):
    (m:Material {uid, formula, formation_energy_per_atom, num_atoms,
                 num_edges, graph_index, source_id})
    (m)-[:CONTAINS_ELEMENT]->(e:Element {symbol})
    (m)-[:HAS_STRUCTURE]->(st:Structure {lattice_a/b/c, alpha, beta, gamma,
                                         volume, density, num_atoms, num_edges})
    (m)-[:FROM_SOURCE]->(s:Source {name})
    (m)-[:HAS_FORMULA]->(f:Formula {value})
    (m)-[:HAS_PROPERTY]->(p:PropertyMeasurement {name, value, unit, source})

All public methods return plain JSON-serializable Python objects so they can
be handed directly to an MCP tool response or any other consumer.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env from the project root (two levels up: research/phase_2 -> root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_URI = os.getenv("NEO4J_URI")
DEFAULT_USER = os.getenv("NEO4J_USERNAME")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE")


class MaterialsRetriever:
    """Thin, read-only Neo4j client for formation-energy-driven retrieval."""

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "MaterialsRetriever":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── internal helper ──────────────────────────────────────────────────────
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def verify_connectivity(self) -> dict[str, Any]:
        """Ping the database and report basic counts."""
        self._driver.verify_connectivity()
        rows = self._run(
            """
            MATCH (m:Material)
            RETURN count(m) AS material_count,
                   min(m.formation_energy_per_atom) AS min_energy,
                   max(m.formation_energy_per_atom) AS max_energy
            """
        )
        return rows[0] if rows else {"material_count": 0}

    # ── core inverse-design query ────────────────────────────────────────────
    def find_by_formation_energy(
        self,
        min_energy: Optional[float] = None,
        max_energy: Optional[float] = None,
        include_elements: Optional[list[str]] = None,
        exclude_elements: Optional[list[str]] = None,
        only_elements: Optional[list[str]] = None,
        limit: int = 20,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """
        Retrieve materials filtered by formation energy and (optionally)
        chemistry.

        Args:
            min_energy: lower bound on formation_energy_per_atom (eV/atom).
            max_energy: upper bound on formation_energy_per_atom (eV/atom).
                        For "low formation energy" requests, set max_energy
                        (e.g. max_energy=-1.0) and leave min_energy=None.
            include_elements: material must contain ALL of these element symbols.
            exclude_elements: material must contain NONE of these symbols.
            only_elements: material's element set must be a SUBSET of these
                           (i.e. restrict to a chemical system, e.g. Li-Fe-O).
            limit: max number of materials to return.
            order: "asc" (most stable first) or "desc".

        Returns:
            List of material dicts with energy, formula, composition, size.
        """
        order_kw = "ASC" if str(order).lower() != "desc" else "DESC"

        where = ["m.formation_energy_per_atom IS NOT NULL"]
        params: dict[str, Any] = {"limit": int(limit)}

        if min_energy is not None:
            where.append("m.formation_energy_per_atom >= $min_energy")
            params["min_energy"] = float(min_energy)
        if max_energy is not None:
            where.append("m.formation_energy_per_atom <= $max_energy")
            params["max_energy"] = float(max_energy)

        if include_elements:
            params["include_elements"] = [s.strip() for s in include_elements]
            where.append(
                "ALL(sym IN $include_elements WHERE "
                "EXISTS { (m)-[:CONTAINS_ELEMENT]->(:Element {symbol: sym}) })"
            )
        if exclude_elements:
            params["exclude_elements"] = [s.strip() for s in exclude_elements]
            where.append(
                "NONE(sym IN $exclude_elements WHERE "
                "EXISTS { (m)-[:CONTAINS_ELEMENT]->(:Element {symbol: sym}) })"
            )

        where_clause = " AND ".join(where)

        only_clause = ""
        if only_elements:
            params["only_elements"] = [s.strip() for s in only_elements]
            only_clause = (
                "WITH m, [(m)-[:CONTAINS_ELEMENT]->(e:Element) | e.symbol] AS syms\n"
                "WHERE ALL(sym IN syms WHERE sym IN $only_elements)\n"
            )

        query = f"""
        MATCH (m:Material)
        WHERE {where_clause}
        {only_clause}
        WITH m, [(m)-[:CONTAINS_ELEMENT]->(e:Element) | e.symbol] AS elements
        RETURN m.uid               AS uid,
               m.formula           AS formula,
               m.formation_energy_per_atom AS formation_energy_per_atom,
               m.num_atoms         AS num_atoms,
               m.num_edges         AS num_edges,
               m.graph_index       AS graph_index,
               apoc.coll.sort(elements) AS elements
        ORDER BY m.formation_energy_per_atom {order_kw}
        LIMIT $limit
        """
        # apoc may not be installed; fall back to plain sort if it fails.
        try:
            return self._run(query, **params)
        except Exception:
            query = query.replace("apoc.coll.sort(elements)", "elements")
            return self._run(query, **params)

    # ── interpolation-parent retrieval (feeds the GNN pipeline) ──────────────
    def find_interpolation_parents(
        self,
        target_energy: float,
        tolerance: float = 0.2,
        include_elements: Optional[list[str]] = None,
        max_buckets: int = 10,
        max_per_bucket: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Find groups of materials that are *compatible for graph interpolation*
        (identical num_atoms AND num_edges) and whose formation energies sit
        within +/- tolerance of a target energy.

        This directly supports the inverse-design loop: pick two parents from
        the same bucket, interpolate their bond distances (see
        test_new_material.py), and screen the candidate with the GNN.

        Returns:
            List of buckets, each: {num_atoms, num_edges, count, members[]}.
        """
        params: dict[str, Any] = {
            "low": float(target_energy) - float(tolerance),
            "high": float(target_energy) + float(tolerance),
            "max_buckets": int(max_buckets),
            "max_per_bucket": int(max_per_bucket),
        }

        elem_clause = ""
        if include_elements:
            params["include_elements"] = [s.strip() for s in include_elements]
            elem_clause = (
                "AND ALL(sym IN $include_elements WHERE "
                "EXISTS { (m)-[:CONTAINS_ELEMENT]->(:Element {symbol: sym}) })"
            )

        query = f"""
        MATCH (m:Material)
        WHERE m.formation_energy_per_atom IS NOT NULL
          AND m.formation_energy_per_atom >= $low
          AND m.formation_energy_per_atom <= $high
          AND m.num_atoms IS NOT NULL
          AND m.num_edges IS NOT NULL
          {elem_clause}
        WITH m.num_atoms AS num_atoms, m.num_edges AS num_edges,
             collect({{
                 uid: m.uid,
                 formula: m.formula,
                 formation_energy_per_atom: m.formation_energy_per_atom,
                 graph_index: m.graph_index
             }})[0..$max_per_bucket] AS members,
             count(*) AS count
        WHERE count >= 2
        RETURN num_atoms, num_edges, count, members
        ORDER BY count DESC
        LIMIT $max_buckets
        """
        return self._run(query, **params)

    # ── structural neighbours of a seed material ─────────────────────────────
    def find_similar_structures(
        self,
        uid: str,
        energy_tolerance: Optional[float] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Given a seed material uid, find other materials with the SAME
        (num_atoms, num_edges) so they can be interpolated with the seed.
        Optionally restrict candidates to within energy_tolerance of the seed.
        """
        params: dict[str, Any] = {"uid": uid, "limit": int(limit)}
        energy_clause = ""
        if energy_tolerance is not None:
            params["tol"] = float(energy_tolerance)
            energy_clause = (
                "AND other.formation_energy_per_atom IS NOT NULL\n"
                "AND abs(other.formation_energy_per_atom - "
                "seed.formation_energy_per_atom) <= $tol"
            )

        query = f"""
        MATCH (seed:Material {{uid: $uid}})
        MATCH (other:Material)
        WHERE other.uid <> seed.uid
          AND other.num_atoms = seed.num_atoms
          AND other.num_edges = seed.num_edges
          {energy_clause}
        WITH seed, other,
             [(other)-[:CONTAINS_ELEMENT]->(e:Element) | e.symbol] AS elements
        RETURN other.uid     AS uid,
               other.formula AS formula,
               other.formation_energy_per_atom AS formation_energy_per_atom,
               other.num_atoms AS num_atoms,
               other.num_edges AS num_edges,
               other.graph_index AS graph_index,
               elements,
               abs(other.formation_energy_per_atom -
                   seed.formation_energy_per_atom) AS energy_distance
        ORDER BY energy_distance ASC
        LIMIT $limit
        """
        return self._run(query, **params)

    # ── full detail lookup ───────────────────────────────────────────────────
    def get_material(self, uid: str) -> Optional[dict[str, Any]]:
        """Return full details for a single material (incl. structure)."""
        rows = self._run(
            """
            MATCH (m:Material {uid: $uid})
            OPTIONAL MATCH (m)-[:HAS_STRUCTURE]->(st:Structure)
            OPTIONAL MATCH (m)-[:FROM_SOURCE]->(s:Source)
            WITH m, st, s,
                 [(m)-[:CONTAINS_ELEMENT]->(e:Element) | e.symbol] AS elements
            RETURN m.uid AS uid,
                   m.formula AS formula,
                   m.formation_energy_per_atom AS formation_energy_per_atom,
                   m.num_atoms AS num_atoms,
                   m.num_edges AS num_edges,
                   m.graph_index AS graph_index,
                   m.source_id AS source_id,
                   s.name AS source,
                   elements,
                   st {
                       .lattice_a, .lattice_b, .lattice_c,
                       .alpha, .beta, .gamma, .volume, .density
                   } AS structure
            """,
            uid=uid,
        )
        return rows[0] if rows else None

    # ── distribution statistics (help users pick a range) ────────────────────
    def energy_statistics(
        self, include_elements: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Summary statistics of formation_energy_per_atom across the (optionally
        element-filtered) dataset, including percentiles. Useful for deciding
        what "low" means before issuing a range query.
        """
        params: dict[str, Any] = {}
        elem_clause = ""
        if include_elements:
            params["include_elements"] = [s.strip() for s in include_elements]
            elem_clause = (
                "AND ALL(sym IN $include_elements WHERE "
                "EXISTS { (m)-[:CONTAINS_ELEMENT]->(:Element {symbol: sym}) })"
            )

        query = f"""
        MATCH (m:Material)
        WHERE m.formation_energy_per_atom IS NOT NULL
        {elem_clause}
        RETURN count(*) AS count,
               min(m.formation_energy_per_atom) AS min,
               max(m.formation_energy_per_atom) AS max,
               avg(m.formation_energy_per_atom) AS mean,
               percentileCont(m.formation_energy_per_atom, 0.05) AS p05,
               percentileCont(m.formation_energy_per_atom, 0.25) AS p25,
               percentileCont(m.formation_energy_per_atom, 0.50) AS median,
               percentileCont(m.formation_energy_per_atom, 0.75) AS p75,
               percentileCont(m.formation_energy_per_atom, 0.95) AS p95
        """
        rows = self._run(query, **params)
        return rows[0] if rows else {"count": 0}


if __name__ == "__main__":
    # Quick manual smoke test.
    with MaterialsRetriever() as r:
        print("Connectivity:", r.verify_connectivity())
        print("Stats:", r.energy_statistics())
        print("Most stable 5:", r.find_by_formation_energy(limit=5))
