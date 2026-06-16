"""
Neo4j MCP server for materials inverse design.

Exposes the formation-energy retrieval layer (retrieval.py) as MCP tools so an
LLM agent can search the materials knowledge graph during an inverse-design
loop, e.g.:

    "Find candidate materials with low formation energy (< -2 eV/atom)
     containing oxygen, then give me interpolation parents near -2.5 eV/atom."

Run (stdio transport, for Cursor / Claude Desktop):
    python neo4j_mcp.py

Connection settings are read from the project-root .env:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from retrieval import MaterialsRetriever
mcp = FastMCP("thanhhoa-materials", host="127.0.0.1", port=8765)

# A single long-lived retriever (driver manages its own connection pool).
_retriever: Optional[MaterialsRetriever] = None


def _get() -> MaterialsRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MaterialsRetriever()
    return _retriever


@mcp.tool()
def check_connection() -> dict[str, Any]:
    """Verify the Neo4j connection and report material count + energy range.

    Call this first to confirm the knowledge graph is reachable.
    """
    return _get().verify_connectivity()


@mcp.tool()
def energy_statistics(include_elements: Optional[list[str]] = None) -> dict[str, Any]:
    """Get formation-energy distribution stats (min/max/mean/percentiles).

    Use this before a range query to decide what "low" formation energy means
    for the dataset (or for a specific chemical subset).

    Args:
        include_elements: optional list of element symbols (e.g. ["O", "Fe"]);
            restricts statistics to materials containing ALL of them.
    """
    return _get().energy_statistics(include_elements=include_elements)


@mcp.tool()
def find_low_formation_energy_materials(
    max_energy: Optional[float] = None,
    min_energy: Optional[float] = None,
    include_elements: Optional[list[str]] = None,
    exclude_elements: Optional[list[str]] = None,
    only_elements: Optional[list[str]] = None,
    limit: int = 20,
    order: str = "asc",
) -> list[dict[str, Any]]:
    """Retrieve materials by formation energy for inverse design.

    Formation energy is in eV/atom; lower (more negative) = more stable.

    Typical inverse-design uses:
      - "low formation energy": set max_energy (e.g. max_energy=-1.5),
        leave min_energy=None, order="asc" (most stable first).
      - "formation energy in a range": set both min_energy and max_energy
        (e.g. min_energy=-2.5, max_energy=-2.0).

    Args:
        max_energy: upper bound on formation_energy_per_atom (eV/atom).
        min_energy: lower bound on formation_energy_per_atom (eV/atom).
        include_elements: material must contain ALL of these element symbols.
        exclude_elements: material must contain NONE of these symbols.
        only_elements: restrict to a chemical system; the material's element
            set must be a SUBSET of this list (e.g. ["Li", "Fe", "O"]).
        limit: maximum number of materials to return (default 20).
        order: "asc" for most stable first, "desc" for least stable.
    """
    return _get().find_by_formation_energy(
        min_energy=min_energy,
        max_energy=max_energy,
        include_elements=include_elements,
        exclude_elements=exclude_elements,
        only_elements=only_elements,
        limit=limit,
        order=order,
    )


@mcp.tool()
def find_interpolation_parents(
    target_energy: float,
    tolerance: float = 0.2,
    include_elements: Optional[list[str]] = None,
    max_buckets: int = 10,
    max_per_bucket: int = 8,
) -> list[dict[str, Any]]:
    """Find graph-compatible parent materials for generating a new candidate.

    Returns buckets of materials that share identical (num_atoms, num_edges)
    and whose formation energies fall within +/- tolerance of target_energy.
    Two materials from the same bucket can be interpolated (bond-distance
    blend, see test_new_material.py) to synthesize a new candidate whose
    expected formation energy is near target_energy.

    Args:
        target_energy: desired formation energy of the new material (eV/atom).
        tolerance: half-width of the energy window around target (eV/atom).
        include_elements: require ALL of these element symbols in parents.
        max_buckets: max number of (num_atoms, num_edges) groups to return.
        max_per_bucket: max member materials listed per bucket.
    """
    return _get().find_interpolation_parents(
        target_energy=target_energy,
        tolerance=tolerance,
        include_elements=include_elements,
        max_buckets=max_buckets,
        max_per_bucket=max_per_bucket,
    )


@mcp.tool()
def find_similar_structures(
    uid: str,
    energy_tolerance: Optional[float] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find materials structurally compatible with a given seed material.

    Returns materials sharing the seed's (num_atoms, num_edges) — i.e. valid
    interpolation partners for that specific seed — sorted by closeness in
    formation energy.

    Args:
        uid: the seed material's uid (e.g. "MP_mp-1103373").
        energy_tolerance: if set, only return partners whose formation energy
            is within this many eV/atom of the seed.
        limit: maximum number of partners to return.
    """
    return _get().find_similar_structures(
        uid=uid, energy_tolerance=energy_tolerance, limit=limit
    )


@mcp.tool()
def get_material(uid: str) -> dict[str, Any]:
    """Get full details for one material, including its crystal structure.

    Args:
        uid: the material's uid (e.g. "MP_mp-1103373").
    """
    result = _get().get_material(uid)
    return result if result is not None else {"error": f"Material '{uid}' not found"}


if __name__ == "__main__":
    import sys

    # Cursor launches this via mcp.json with stdin as a pipe (not a TTY).
    # Running manually from a terminal means stdin IS a TTY → start an HTTP
    # server so LM Studio can reach it at http://127.0.0.1:8765/mcp.
    if sys.stdin.isatty():
        # Manual run: HTTP transport for LM Studio
        mcp.run(transport="streamable-http")
    else:
        # Cursor-managed: stdio transport
        mcp.run()
