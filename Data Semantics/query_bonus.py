import rdflib

# ─────────────────────────────────────────────────────────────────
# ECS7028U/P Data Semantics - query_bonus.py
# Intermediate Task: Query fused ontology (DBpedia + Wikidata)
#
# Run AFTER bonus.py has created football_full.owl
# Usage: python query_bonus.py
#
# These queries answer questions that NEITHER DBpedia NOR Wikidata
# could answer alone — demonstrating successful data fusion.
# ─────────────────────────────────────────────────────────────────

g = rdflib.Graph()
g.parse("football_full.owl", format="xml")
print(f"Graph loaded: {len(g)} triples\n")

# ── Fusion Query 1: English Players with Market Value > 50M ─────
# DBpedia provided: player-team relationships and nationality
# Wikidata provided: market values
# Neither alone could answer this!
print("=" * 60)
print("FUSION Q1: English Players with Market Value > 50 Million")
print("(Combines DBpedia nationality + Wikidata market values)")
print("=" * 60)

q1 = """
PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?playerName ?teamName ?marketValue ?countryName
WHERE {
    ?player rdf:type fb:Player .
    ?player fb:hasName ?playerName .
    ?player fb:hasMarketValue ?marketValue .
    ?player fb:hasNationality ?country .
    ?country fb:hasName ?countryName .
    OPTIONAL {
        ?player fb:playsFor ?team .
        ?team fb:hasName ?teamName
    }
    FILTER (CONTAINS(LCASE(str(?countryName)), "england")
            || CONTAINS(LCASE(str(?countryName)), "united kingdom"))
    FILTER (?marketValue > 50000000)
}
ORDER BY DESC(?marketValue)
"""

print(f"{'Player':<28} {'Team':<28} {'Value (EUR)':<15} {'Country'}")
print("-" * 85)
results = list(g.query(q1))
if results:
    for row in results:
        player  = str(row[0]) if row[0] else "N/A"
        team    = str(row[1]) if row[1] else "N/A"
        value   = str(row[2]) if row[2] else "N/A"
        country = str(row[3]) if row[3] else "N/A"
        print(f"{player:<28} {team:<28} {value:<15} {country}")
else:
    print("No results — try lowering the market value threshold.")

# ── Fusion Query 2: Teams with Stadium Capacity > 50000 ─────────
# DBpedia: stadium capacity
# Wikidata: player market values at that team
print("\n" + "=" * 60)
print("FUSION Q2: High-Capacity Stadiums and Their Top Players")
print("(Combines DBpedia stadium data + Wikidata player values)")
print("=" * 60)

q2 = """
PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?teamName ?stadiumName ?capacity ?playerName ?marketValue
WHERE {
    ?team    rdf:type fb:Team .
    ?team    fb:hasName ?teamName .
    ?team    fb:hasStadium ?stadium .
    ?stadium fb:hasName ?stadiumName .
    ?stadium fb:stadiumCapacity ?capacity .
    FILTER (?capacity > 50000)
    OPTIONAL {
        ?player fb:playsFor ?team .
        ?player fb:hasName ?playerName .
        ?player fb:hasMarketValue ?marketValue
    }
}
ORDER BY DESC(?capacity) DESC(?marketValue)
LIMIT 20
"""

print(f"{'Team':<25} {'Stadium':<25} {'Capacity':<12} {'Player':<25} {'Value'}")
print("-" * 95)
results2 = list(g.query(q2))
if results2:
    for row in results2:
        team     = str(row[0]) if row[0] else "N/A"
        stadium  = str(row[1]) if row[1] else "N/A"
        capacity = str(row[2]) if row[2] else "N/A"
        player   = str(row[3]) if row[3] else "N/A"
        value    = str(row[4]) if row[4] else "N/A"
        print(f"{team:<25} {stadium:<25} {capacity:<12} {player:<25} {value}")
else:
    print("No results found.")

print("\nThese results combine data from TWO sources:")
print("  - DBpedia  : team, stadium, capacity, nationality")
print("  - Wikidata : player market values")
print("Neither source alone could answer these queries!")
