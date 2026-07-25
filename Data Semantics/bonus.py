import rdflib
from rdflib import Graph, URIRef
from SPARQLWrapper import SPARQLWrapper, XML

# ─────────────────────────────────────────────────────────────────
# ECS7028U/P Data Semantics - bonus.py
# Football Ontology - Data Fusion from Wikidata (Second Source)
#
# Intermediate Task: This script pulls additional player market
# value and nationality data from Wikidata — information that
# DBpedia alone cannot provide. It merges this with the existing
# football_basic.owl to answer queries neither source could
# answer alone (e.g. "Which English players are worth over 50M?")
# ─────────────────────────────────────────────────────────────────

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader(
    "User-Agent",
    "FootballOntologyProject/1.0 (ECS7028 coursework; student@qmul.ac.uk)"
)

construct_query = """
    PREFIX fb:   <http://www.semanticweb.org/ontologies/football#>
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX wd:   <http://www.wikidata.org/entity/>
    PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

    CONSTRUCT {
        ?player  rdf:type fb:Player .
        ?player  fb:hasName ?playerName .
        ?player  fb:hasMarketValue ?marketValue .
        ?player  fb:hasNationality ?country .
        ?country rdf:type fb:Country .
        ?country fb:hasName ?countryName .
        ?team    rdf:type fb:Team .
        ?team    fb:hasName ?teamName .
        ?player  fb:playsFor ?team
    }
    WHERE {
        ?player wdt:P31 wd:Q5 .
        ?player wdt:P106 wd:Q937857 .
        ?player wdt:P54 ?team .
        ?team   wdt:P118 wd:Q9448 .
        ?player rdfs:label ?playerName .
        FILTER (LANG(?playerName) = 'en')
        ?team   rdfs:label ?teamName .
        FILTER (LANG(?teamName) = 'en')

        OPTIONAL {
            ?player wdt:P2218 ?marketValue .
        }
        OPTIONAL {
            ?player wdt:P27 ?country .
            ?country rdfs:label ?countryName .
            FILTER (LANG(?countryName) = 'en')
        }
    }
    LIMIT 150
"""

sparql.setQuery(construct_query)
sparql.setReturnFormat(XML)

print("Querying Wikidata for player market values...")
print("(This may take 30-60 seconds...)")

# Run query
g = sparql.query().convert()

# Merge with basic ontology (which already has DBpedia data)
g.parse("football_basic.owl", format="xml")

# Save fused ontology
g.serialize("football_full.owl", format="xml")

print(f"\nDone! Graph has {len(g)} triples.")
print("Saved as: football_full.owl")
print("\nThis file contains data from BOTH DBpedia AND Wikidata.")
print("Open football_full.owl in Protege to inspect all individuals.")
