import rdflib

# ─────────────────────────────────────────────────────────────────
# ECS7028U/P Data Semantics - query_basic.py
# Query the local football ontology (football_basic.owl)
#
# Run AFTER basic.py has created football_basic.owl
# Usage: python query_basic.py
# ─────────────────────────────────────────────────────────────────

g = rdflib.Graph()
g.parse("football_basic.owl", format="xml")
print(f"Graph loaded: {len(g)} triples\n")

# ── Query 1: All Teams and their Stadiums ────────────────────────
print("=" * 55)
print("Q1: Premier League Teams and Their Stadiums")
print("=" * 55)

q1 = """
PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?teamName ?stadiumName ?capacity
WHERE {
    ?team    rdf:type fb:Team .
    ?team    fb:hasName ?teamName .
    OPTIONAL {
        ?team    fb:hasStadium ?stadium .
        ?stadium fb:hasName ?stadiumName .
        OPTIONAL { ?stadium fb:stadiumCapacity ?capacity }
    }
}
ORDER BY ?teamName
"""

print(f"{'Team':<35} {'Stadium':<35} {'Capacity'}")
print("-" * 80)
for row in g.query(q1):
    team     = str(row[0]) if row[0] else "N/A"
    stadium  = str(row[1]) if row[1] else "N/A"
    capacity = str(row[2]) if row[2] else "N/A"
    print(f"{team:<35} {stadium:<35} {capacity}")

# ── Query 2: Players and their Teams ────────────────────────────
print("\n" + "=" * 55)
print("Q2: Players and Their Teams")
print("=" * 55)

q2 = """
PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?playerName ?teamName ?countryName
WHERE {
    ?player rdf:type fb:Player .
    ?player fb:hasName ?playerName .
    OPTIONAL {
        ?player fb:playsFor ?team .
        ?team   fb:hasName ?teamName
    }
    OPTIONAL {
        ?player fb:hasNationality ?country .
        ?country fb:hasName ?countryName
    }
}
ORDER BY ?teamName ?playerName
LIMIT 30
"""

print(f"{'Player':<30} {'Team':<30} {'Nationality'}")
print("-" * 75)
for row in g.query(q2):
    player  = str(row[0]) if row[0] else "N/A"
    team    = str(row[1]) if row[1] else "N/A"
    country = str(row[2]) if row[2] else "N/A"
    print(f"{player:<30} {team:<30} {country}")

# ── Query 3: Teams founded before 1900 ──────────────────────────
print("\n" + "=" * 55)
print("Q3: Teams Founded Before 1900")
print("=" * 55)

q3 = """
PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?teamName ?founded
WHERE {
    ?team rdf:type fb:Team .
    ?team fb:hasName ?teamName .
    ?team fb:foundedYear ?founded .
    FILTER (?founded < 1900)
}
ORDER BY ?founded
"""

print(f"{'Team':<35} {'Founded'}")
print("-" * 45)
for row in g.query(q3):
    print(f"{str(row[0]):<35} {str(row[1])}")

print("\nDone! Try editing the queries above to explore the ontology.")
