User journey
============

As an administrator, I want to set up a collection of documents and claims around a research question.
------------------------------------------------------------------------------------------------------

#. Create the collection
    #. Home page
        .. figure:: screenshots/collection_list.png

            `collection list`_

    #. Add collection section / Submit
#. Configure the collection
    .. figure:: screenshots/collection_edit.png

        `collection edit`_

    #. Choose active embeddings
#. Add documents and claims from another collection (TODO)
#. Manage user rights for the collection
    #. Menu *Admin*
        .. figure:: screenshots/collection_admin.png

            `collection admin`_

        Note that rights provided at the global level (`general admin`_) cannot be taken away.

As a collection curator, I want to add seed statements to the collection
------------------------------------------------------------------------

#. Start from the collection list in the home page
    .. figure:: screenshots/collection_list.png

        `collection list`_

#. Click on the collection name to get the base collection view
    .. figure:: screenshots/collection_view.png

        `collection view`_

    You may return to the collection view at any time by clicking on the logo.

#. Follow the link to *Propose a new claim*
    .. figure:: screenshots/propose_statement_1.png

        `propose statement`_

#. Enter the statement text, click the *search* button.
    .. figure:: screenshots/propose_statement_2.png

        `propose statement 2`_

    This step allows you to verify whether a similar or related statement already exists.

#. Set the statement subtype (e.g. Claim, Question, etc.) and confirm that you want to add the new statement.

#. Alternatively: choose a related statement and click the *Relate to* button.
    .. figure:: screenshots/propose_statement_3.png

        `propose related statement`_

    In that case, you would not only set the new statement's subtype but also the relation to the other claim.


As the collection administrator, I want to choose one of the questions as the root theme of the collection
----------------------------------------------------------------------------------------------------------

TODO

As a collection curator, I want to add documents to the collection for research
-------------------------------------------------------------------------------

#. From the `collection view`_, follow the link to *Upload documents*.
    .. figure:: screenshots/document_upload.png

        `document upload`_

#. I want to tell ClaimMiner to download a URL:
    Set the URL the first URL field and press submit.

    TODO: Redirect to document.

#. I want to upload a local document to ClaimMiner:
    Upload with the *upload* button. Set a URI as a unique identifier (e.g.: DOI.) Press submit.

#. I want to tell ClaimMiner to download many URLs:
    Upload a CSV file in the second form. Tell ClaimMiner in which column to look for URLs (if not the first.) Also specify whether to skip the first (title) row.


As a collection curator, I want to see statements extracted from a specific document
------------------------------------------------------------------------------------

    .. figure:: screenshots/document_view.png

        `document view`_

As a collection curator, I want to identify quotes relevant to a statement (automatic)
--------------------------------------------------------------------------------------

#. Search for the base statement itself
    #. Menu: *Search*
        .. figure:: screenshots/text_based_search.png

            `text based search`_

    #. Choose text or semantic search (and model in the latter case)
    #. Select *claims search* and deselect *paragraph* search
    #. Click on search to get results
    #. Click on statement link to the statement view
        .. figure:: screenshots/statement_view.png

            `statement view`_

#. Click on the Run button of the "infer_quotes_proximity - base_infer_quotes_proximity" task.
#. Wait for results, and go back to the statement view. Reload the page.


As a collection curator, I want to search for new documents related to a generated graph
----------------------------------------------------------------------------------------

#. Navigate to the base statement as above
#. Click *Run* on *autosearch* task
#. Wait, possibly a full day...
#. New documents will be ingested, it is worth re-running the task to find quotes.


As a collection curator, I want to identify quotes relevant to a statement (manual, incomplete)
-----------------------------------------------------------------------------------------------

TODO: Rebuild around Structured Ideas

#. Search for the base statement itself
    #. Menu: *Search*
        .. figure:: screenshots/text_based_search.png

            `text based search`_

    #. Choose text or semantic search (and model in the latter case)
    #. Select *claims search* and deselect *paragraph* search
    #. Click on search to get results
    #. Click on statement link to the statement view
        .. figure:: screenshots/statement_view.png

            `statement view`_

#. Click the *Claim Search* link from the statement view to look at surrounding paragraphs
    .. figure:: screenshots/statement_based_search_top.png

        `statement based search`_

#. Add as quote (TODO)


As a collection curator, I want to extract claims from paragraphs related to an existing statement (Using prompts)
------------------------------------------------------------------------------------------------------------------

For example, the root statement might be a question and we want to extract positions, or go from positions to arguments.

#. As above, look at paragraphs around a statement.
#. Select a few paragraphs that seem likely to contain related claims using the checkboxes
    #. Note that you can go to next/previous page and the selection will be not be lost
#. Choose a paragraph-based prompt (eg *question_to_position*) in the form at the bottom of the page
    .. figure:: screenshots/statement_based_search_bottom.png

        `statement based search`_

#. Click the *LLM analysis* button

    .. figure:: screenshots/prompt_analysis_results.png

        `prompt analysis results`_

#. Save relevant results as Statement (*Save* button)
    * Note: The prompt used to keep related quotes, this seems broken, probably due to changes in LLM?

As a curator, I want to extract derived statements from a statement
-------------------------------------------------------------------

For example, transform a question into its affirmative forms, or generate answers without reference to documents.

#. Go to the `statement view`_ screen as above
#. Choose a simple (not paragraph-based) prompt (eg *find_implicit*) and click the *LLM analysis* button
#. Save relevant results as Statement (using the *Save* button)


As a curator, I want to cluster similar statements to diminish redundancy
-------------------------------------------------------------------------

#. Look at existing stored cluster computation analyses
    #. Menu *Utilities / Claim clusters*
        .. figure:: screenshots/cluster_analysis_list.png

            `cluster analysis list`_

#. Compute clusters on the collection (eg if nothing had been stored)
    #. Click the link to *New cluster calculation*
        .. figure:: screenshots/new_clusters.png

            `new clusters`_

    #. Play with parameters and use the *recalculate* button if the resulting clusters are too broad or too narrow
    #. Save the cluster computation analysis (*save* button)
#. Look at a stored cluster computation analysis
    #. From `cluster analysis list`_, link to id of specific cluster analysis
        .. figure:: screenshots/cluster_list.png

            `cluster list`_

#. Edit a specific cluster:
    #. Link to id of cluster
        .. figure:: screenshots/cluster_details.png

            `cluster details`_

    #. Add neighbours to cluster (+ icon, ->+ needs repairs)
    #. Remove neighbours from cluster (trash icon)
    #. Validate neighbours in cluster (checkmark icon)
    #. Invalidate statements in cluster (TODO)
    #. Make a statement the basis of a new cluster (⨁)
    #. Make a statement the central statement of a cluster (🎯)

As a collection curator, I want to organize Statements into Ideas
-----------------------------------------------------------------

TODO

As a collection curator, I want to search for new documents related to a Statement (GDelt)
------------------------------------------------------------------------------------------

Warning: There is a significant cost per search.

#. Navigate to `statement view`_ as above
#. Click *GDelt search* link
    .. figure:: screenshots/gdelt_search.png

        `gdelt search`_

#. Choose a limit for number of documents, most recent date.
    * TODO: Auto-update with last search
    * TODO: maybe a maximum distance?
#. Add documents.
    * TODO: I should record which documents came from GDELT queries.


As a collection curator, I want to search for new documents related to a single Statement (Junto, untested)
-----------------------------------------------------------------------------------------------------------

#. Navigate to `statement view`_ as above
#. Click *Run* on *fact_check_claim* task


As a collection curator, I want to upload of many statements in bulk
--------------------------------------------------------------------

#.  From statement list page, *Bulk upload* link
    .. figure:: screenshots/statement_upload.png

    `statement upload`_

#. Choose the statement subtype and the CSV column.

As a collection curator, I want to browse for all the available statements and documents to get an overview of what is there
----------------------------------------------------------------------------------------------------------------------------

#. Claim list (Menu: *Claims*)

        .. figure:: screenshots/statement_list.png

            `statement list`_

        This page allows you to browse existing statements (in alphabetical order.)

#. Document list (Menu: *Document*)

        .. figure:: screenshots/document_list.png

            `document list`_

        This page allows you to browse existing documents (in reverse order of entry)



As an administrator, I want to create a new AI prompt (or edit an existing draft prompt)
----------------------------------------------------------------------------------------

#.  Menu *Utilities / Prompt*
    .. figure:: screenshots/prompt_list.png

    `prompt list`_

#.  Click on *Add prompt* button (Or click on link to existing prompt name)
    .. figure:: screenshots/edit_prompt.png

    `edit prompt`_

#.  Edit parameters until satisfied, then uncheck "keep editing" and press "Ok"
#.  You can in theory edit a non-draft prompt, but then you have to delete the generated prompts. Bad idea.
#.  TODO: Create a clone, mark original prompt as obsolete in that case.

As a system administrator, I want to manage users
-------------------------------------------------

#. Menu *Admin* (from base URL, i.e. not in a collection)
    .. figure:: screenshots/general_admin.png

        `general admin`_

    Use it to confirm users who created an account, and give them permissions across all collections


Remaining screens
=================

#. General dashboard
    #. https://claimminer.conversence.com/dashboard
        .. figure:: screenshots/dashboard.png

            `dashboard`_

#. Cloud
    #. Menu *Utilities / Cloud*
        .. figure:: screenshots/statement_scatter_plot.png

            `statement scatter plot`_



.. _cluster analysis list: https://claimminer.conversence.com/c/agi3/claim/clusters
.. _cluster details: https://claimminer.conversence.com/c/agi3/claim/clusters/823268/823420
.. _cluster list: https://claimminer.conversence.com/c/agi3/claim/clusters/823268
.. _collection admin: https://claimminer.conversence.com/c/agi/admin
.. _collection edit: https://claimminer.conversence.com/c/agi3/edit
.. _collection list: https://claimminer.conversence.com/
.. _collection view: https://claimminer.conversence.com/c/agi3
.. _dashboard: https://claimminer.conversence.com/dashboard
.. _document list: https://claimminer.conversence.com/c/agi3/document
.. _document upload: https://claimminer.conversence.com/c/agi3/document/upload
.. _document view: https://claimminer.conversence.com/c/agi3/document/692978
.. _edit prompt: https://claimminer.conversence.com/prompt/find_implicit
.. _gdelt search: https://claimminer.conversence.com/c/agi3/claim/857659/gdelt
.. _general admin: https://claimminer.conversence.com/admin
.. _new clusters: https://claimminer.conversence.com/c/agi3/claim/clusters/new
.. _prompt analysis results: https://claimminer.conversence.com/c/agi3/analysis/857893
.. _prompt list: https://claimminer.conversence.com/prompt
.. _propose related statement: https://claimminer.conversence.com/c/agi3/claim/804787/add_related
.. _propose statement 2: https://claimminer.conversence.com/c/agi3/claim/propose
.. _propose statement: https://claimminer.conversence.com/c/agi3/claim/propose
.. _root statements: https://claimminer.conversence.com/c/diablo_canyon/claim_index
.. _statement based search: https://claimminer.conversence.com/c/agi3/claim/857659/search
.. _statement list: https://claimminer.conversence.com/c/agi/claim
.. _statement scatter plot: https://claimminer.conversence.com/c/diablo_canyon/claim/scatter
.. _statement upload: https://claimminer.conversence.com/c/agi3/claim/upload
.. _statement view: https://claimminer.conversence.com/c/agi3/claim/857659
.. _text based search: https://claimminer.conversence.com/c/agi3/search

.. _after propose question: https://claimminer.conversence.com/c/agi3/claim/3277956
