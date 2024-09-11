Installation
============

ClaimMiner has been installed on Ubuntu and MacOS. Familiarity with most common components and installation procedures is assumed.

Prerequisites: Ubuntu Jammy or Noble
------------------------------------

Note: Add instructions so deadsnakes and pgdg repositories are in `/etc/apt/sources.list.d`

.. code-block:: shell-session

    sudo apt install python3.11-dev postgresql-server-dev-16 memcached redis librdkafka-dev nginx graphviz-dev postgresql-plpython3-16 openjdk-11-jdk
    curl -LsSf https://astral.sh/uv/install.sh | sh

Notes
.....

* Python 3.11 is assumed. On ubuntu, it means adding `ppa:deadsnakes/ppa` to `apt`. (`Instructions <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>`_)
* Postgresql 16 is assumed, but 14 or 15 also work. On ubuntu, it means adding the pgdg repository. (`Instructions <https://www.postgresql.org/download/linux/ubuntu/>`_)

Kafka
.....

Kafka is installed by hand, you could follow `these instructions <https://www.conduktor.io/kafka/how-to-install-apache-kafka-on-linux-without-zookeeper-kraft-mode/>`_
A few caveats: In `kafka/config/kraft/erver.properties`, set `log.dirs=/var/lib/kafka/kraft-combined-logs` instead of `/tmp/kraft-combined-logs`, so the logs survive a reboot. I also create a service, setting `/etc/systemd/system/kafka.service` to

.. code-block:: ini

    [Unit]
    Description=kafka.service

    [Service]
    Environment=LC_CTYPE=C.UTF-8
    Environment=KAFKA_PATH=/home/kafka/kafka
    User=kafka
    Group=kafka

    Type=simple
    Restart=on-failure
    ExecStart=/home/kafka/kafka/bin/kafka-server-start.sh /home/kafka/kafka/config/server.properties
    ExecStop=/home/kafka/kafka/bin/kafka-server-stop.sh /home/kafka/kafka/config/server.properties
    KillMode=control-group
    TimeoutStopSec=120

    [Install]
    WantedBy=multi-user.target

(Of course, replace `/home/kafka/kafka` with the actual path to the kafka installation.)

Chrome
......

Install Chrome as described `here <https://askubuntu.com/questions/510056/how-to-install-google-chrome>`_.


Prerequisites: Mac
------------------

.. code-block:: shell-session

    brew install python@3.11 postgresql@14 redis memcached kafka md5sha1sum graphviz numpy uv
    brew tap conversence/ConversenceTaps
    brew install postgresql_plpy@14
    brew services start postgresql@14
    brew services start redis
    brew services start memcached
    brew services start zookeeper
    brew services start kafka

Notes on prerequisites
----------------------

* You can find mac wheels for tensorflow-text `here <https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases>`_.
* You can find a mac wheel for pygraphviz `here <http://idealoom.org/wheelhouse/pygraphviz-1.12-cp311-cp311-macosx_14_0_arm64.whl>`
* Postgres 16 is used on Ubuntu and mac, but on mac, the homebrew postgres 16 recipe does not handle extensions as well as 14.
* It is simpler to operate Kafka in KRaft mode, you don't need zookeeper. But on mac, it's simpler to follow the default with zookeeper.
* Nginx is for production, it is not usually installed on mac.

PgVector
--------

Install pgvector from `source <https://github.com/pgvector/pgvector>`_, following instructions. (A traditional ``make ; sudo make install``.)

It may be necessary to set ``PG_CONFIG`` to the appropriate path:
``/usr/lib/postgresql/15/bin/pg_config`` on linux, ``/opt/homebrew/opt/postgresql@14/bin/pg_config`` on mac.

ClaimMiner
----------

1. Clone the repository and ``cd`` into it
2. Use `uv` to create and populate the virtual environment: ``uv sync``
3. Activate the virtual environment: ``. .venv/bin/activate``
4. Create a skeleton config.ini file by calling initial setup. Exact arguments will depend on platform. The point is to pass database administrator credentials.

  1. Ubuntu, assuming a postgres user exists, and the current user is a sudoer:

    1. ``python scripts/initial_setup.py --app_name ClaimMiner --config-template config.ini.tmpl --sudo -u postgres``
    2. Note: I have a non-sudoer user to run ClaimMiner, but login as a sudoer user when necessary for some commands.

  2. Mac, assuming the database accepts the logged-in user as a database admin:

    1. ``python scripts/initial_setup.py --app_name ClaimMiner --config-template config.ini.tmpl``

  3. Note that calls to ``initial_setup.py`` can be repeated without losing information. More options are given in the ``--help``

5. Initialize the development database

  1. ``python scripts/db_updater.py init``
  2. ``python scripts/db_updater.py deploy``

    1. If this fails with permission errors, you may have to do the following: ``chmod -R a+r deploy revert``
    2. Also make sure the whole path is searchable by all

  3. The last command can and should be reapplied to run migrations whenever changes are made to the database schema.
  4. The need to do so can be verified with ``python scripts/db_updater.py status``.
  5. Note: The initial deployment may require a sudoer user on ubuntu.

6. Download model with spaCy ``python -m spacy download en_core_web_sm``
7. Ensure the tensorflow cache is not in temp. (I set ``export TFHUB_CACHE_DIR=$HOME/.cache/tfhub_modules`` in my .bashrc)
8. Install chromedriver using the following stanza in python:

.. code-block:: python

  from selenium import webdriver
  from selenium.webdriver.chrome.service import Service as ChromeService
  from webdriver_manager.chrome import ChromeDriverManager

  driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))


Credentials
-----------

Then, some more credentials need to be added to the ``config.ini``. The following sections or variables need to be added.
(TODO: Add a template for this.)

.. code-block:: ini

    [base]
    google_credentials = <filename>
    spacy_model = en_core_web_sm

    [openai]
    api_key = <key>
    organization = <org_id>

    [web_logging]
    filename = web.log
    level = INFO

    [event_logging]
    filename = events.log
    level = INFO

Here is where and how to obtain each credential:

DebateMap
.........

1. Login with a google account on https://debates.app . Your gmail username will be used by DebateMap.
2. Visit https://debates.app/app-server/gql-playground
3. Use query and data below, then follow instructions from the query results. Record the token.

.. code-block:: gql

    subscription($input: SignInStartInput!) {
      signInStart(input: $input) {
        instructions
        authLink
        resultJWT
      }
    }

.. code-block:: json

    {
      "input":{
      "provider": "google",
      "jwtDuration": 7776000,
      "jwtReadOnly": false,
      "preferredUsername": "<username>"
      }
    }


Google credentials
..................

We use credentials to access GDELT. This is optional.

`Create a project <https://console.cloud.google.com/projectcreate>`_ in the Google console, or reuse one you have; then `create a service account <https://console.cloud.google.com/iam-admin/serviceaccounts>`_ for that project (with Editor role); then create keys for that account (follow the console) and download the key pair as a json file. Place that json file in the file root, and give the filename as credentials.

Then you have to `activate the necessary services <https://console.cloud.google.com/apis/library>`_.
Here is a list of currently activated APIs for the GDELT account (It is possible that all are not necessary...)

* BigQuery API
* BigQuery Reservation API
* BigQuery Storage API
* Cloud Datastore API
* Cloud Debugger API
* Cloud Logging API
* Cloud Monitoring API
* Cloud SQL
* Cloud Storage
* Cloud Storage API
* Cloud Trace API
* Custom Search API
* Google Cloud APIs
* Google Cloud Storage JSON API
* Service Management API
* Service Usage API

You will also have to `define a quota <https://console.cloud.google.com/apis/api/bigquery.googleapis.com/quotas>`_ for the use of BigQuery on the GDELT account. The queries are usually quite expensive, as there is currently no indexing on the embeddings.
We recommend setting the "Extract bytes per day" to 1Gb.

Logging
-------

We provide example files for the logging configuration. To use them, you will want to rename ``web_logging.ini.tmpl`` to ``web_logging.ini`` and ``event_logging.ini.tmpl`` to ``event_logging.ini``, and create the logs directory.


Running (development)
---------------------

In different terminals, where the virtualenv has been activated, run the two following commands:

* ``python -m claim_miner.tasks.kafka``
* ``uvicorn claim_miner.app:app --reload``

Production installation
-----------------------

(To be developed)

* Setup systemd tasks for the web and worker
* Set the ``PRODUCTION=1`` environment variable for the kafka task
* The web task will go through uvicorn: ``<path to venv>/bin/uvicorn claim_miner.app:app``
* Setup a nginx reverse proxy on the uvicorn port. (Select a free port on your machine.)

