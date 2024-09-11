ClaimMiner
==========

Functional overview
-------------------

ClaimMiner was originally designed by Conversence_ for SocietyLibrary_ as a way to identify claim networks in a document corpus.
As such, it offers basic document parsing and RAG functionality, with a plugin architecture to add new analysis tasks.
This open source project represents the core functionality of ClaimMiner, and a lot of analysis tasks specific to Society Library was removed.
We intend to migrate ClaimMiner on HyperKnowledge_ architecture, so expect a lot of changes to the code base in the near future.
In particular, we provisonally removed some integrations that we intend to re-build differently on a HyperKnowledge_ basis, notably DebateMap_ integration, which we remain committed to.

The expected main data flow is as follows:

1. The documents are added to the corpus, either uploaded directly or as URLs
2. URLs are downloaded
3. Documents are broken into paragraphs
4. Language embeddings are calculated for each paragraph
5. Operators input some initial seed claims (or import them from DebateMap)
6. Operators look for semantically related paragraphs in the corpus using the embeddings
7. They send the most promising paragraphs to AI systems that will identify claims in the paragraphs
8. Those claims are vetted, stored in the system and eventually sent to DebateMap.

There are other ancillary functions:

1. ClaimMiner can use GDELT_ to perform a semantic search for news items
2. ClaimMiner can identify clusters in the claims, and draw a cloud of claims
3. ClaimMiner can perform text search on paragraphs or claims
4. ClaimMiner can perform a broadening semantic search (MMR) on paragraphs or claims

Technology stack
----------------

* ClaimMiner's data is mostly stored in Postgres (14 or better). In particular, storing the embeddings requires the use of pgvector_.
* The web server is built using FastAPI_
* It is a classic backend with Jinja_ templates, i.e. not a classic SPA; but we emulate a SPA using htmx_.
* It uses SQLAlchemy_ to talk to the database.
* It uses kafka_ to send work requests to a worker.
* It is served using uvicorn_, through an nginx_ proxy.
* Uploaded or downloaded documents are stored in the file system, and the database keeps a hash reference.
* Some machine learning operations (clustering, tag clouds) are done using scikit-learn_.
* The MMR is computed within the database, with a pl-python_ procedure.
* Database migrations are run using the db_updater_ script.
* Claim identification currently uses OpenAI_ through langchain_.
* Server-side sessions are cached in redis_
* Some more caching is done using memcached_

.. _Postgres: https://www.postgresql.org
.. _DebateMap: https://github.com/debate-map/app
.. _pgvector: https://github.com/pgvector/pgvector
.. _GDELT: https://www.gdeltproject.org/
.. _langchain: https://github.com/hwchase17/langchain
.. _FastAPI: https://fastapi.tiangolo.com
.. _Flask: https://flask.palletsprojects.com/en/
.. _Jinja: https://jinja.palletsprojects.com/en/
.. _SQLAlchemy: https://www.sqlalchemy.org/
.. _htmx: https://htmx.org/
.. _hypercorn: https://pgjones.gitlab.io/hypercorn/
.. _nginx: https://nginx.org
.. _scikit-learn: https://scikit-learn.org/stable/
.. _OpenAI: https://openai.com
.. _pl-python: https://www.postgresql.org/docs/current/plpython.html
.. _kafka: https://kafka.apache.org
.. _redis: https://redis.com
.. _memcached: https://memcached.org
.. _db_updater: db_updater.html
.. _SocietyLibrary: https://www.societylibrary.org
.. _HyperKnowledge: https://hyperknowledge.org
.. _Conversence: https://www.conversence.com
.. _uvicorn: https://www.uvicorn.org
