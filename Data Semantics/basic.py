import rdflib
from rdflib import Graph, URIRef
from SPARQLWrapper import SPARQLWrapper, XML

# ─────────────────────────────────────────────────────────────────
# ECS7028U/P Data Semantics - basic.py
# Football Ontology Population from DBpedia
#
# This script uses a SPARQL CONSTRUCT query to pull data about
# Premier League football teams, players, stadiums, leagues and
# countries from DBpedia, then merges it with our local ontology
# (football.rdf) and saves the result as football_basic.owl
# ─────────────────────────────────────────────────────────────────

sparql = SPARQLWrapper("https://dbpedia.org/sparql")

construct_query = """
    PREFIX fb:  <http://www.semanticweb.org/ontologies/football#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX dbp: <http://dbpedia.org/property/>

    CONSTRUCT {
        ?team    rdf:type fb:Team .
        ?team    fb:hasName ?teamName .
        ?team    fb:foundedYear ?founded .
        ?team    fb:competesin ?league .
        ?league  rdf:type fb:League .
        ?league  fb:hasName ?leagueName .
        ?team    fb:hasStadium ?stadium .
        ?stadium rdf:type fb:Stadium .
        ?stadium fb:hasName ?stadiumName .
        ?stadium fb:stadiumCapacity ?capacity .
        ?team    fb:basedIn ?country .
        ?country rdf:type fb:Country .
        ?country fb:hasName ?countryName .
        ?player  rdf:type fb:Player .
        ?player  fb:hasName ?playerName .
        ?player  fb:hasAge ?age .
        ?player  fb:playsFor ?team .
        ?player  fb:hasNationality ?country
    }
    WHERE {
        ?team dbo:league dbr:Premier_League .
        ?team rdfs:label ?teamName .
        FILTER (LANG(?teamName) = 'en')

        BIND(dbr:Premier_League AS ?league)
        BIND("Premier League"@en AS ?leagueName)

        OPTIONAL {
            ?team dbo:ground ?stadium .
            ?stadium rdfs:label ?stadiumName .
            FILTER (LANG(?stadiumName) = 'en')
            OPTIONAL { ?stadium dbo:seatingCapacity ?capacity }
        }

        OPTIONAL { ?team dbo:foundingYear ?founded }

        OPTIONAL {
            ?team dbo:locationCountry ?country .
            ?country rdfs:label ?countryName .
            FILTER (LANG(?countryName) = 'en')
        }

        OPTIONAL {
            ?player dbo:team ?team .
            ?player rdf:type dbo:SoccerPlayer .
            ?player rdfs:label ?playerName .
            FILTER (LANG(?playerName) = 'en')
            OPTIONAL { ?player dbo:birthYear ?age }
            OPTIONAL {
                ?player dbo:nationality ?country .
                ?country rdfs:label ?countryName .
                FILTER (LANG(?countryName) = 'en')
            }
        }
    }
    LIMIT 200
"""

sparql.setQuery(construct_query)
sparql.setReturnFormat(XML)

print("Querying DBpedia for football data...")
print("(This may take 30-60 seconds...)")

# Run query — returns an RDFLib graph directly (CONSTRUCT)
g = sparql.query().convert()

# Merge with our local ontology T-Box from Protege
g.parse("football.rdf", format="xml")

# Save populated ontology
g.serialize("football_basic.owl", format="xml")

print(f"\nDone! Graph has {len(g)} triples.")
print("Saved as: football_basic.owl")
print("\nYou can open football_basic.owl in Protege to inspect the individuals.")
