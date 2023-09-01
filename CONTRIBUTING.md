# Contributor guidance 

By contributing to PyGOM, you acknowledge that you are assigning copyright to us so we can continue to maintain package. PyGOM is licensed under (GPL2)[https://raw.githubusercontent.com/PublicHealthEngland/pygom/master/LICENSE.txt].

## suggested information for raising issues
- is it possible to produce a template for a new issue?

## requirements for amendments 
- merge and pull requests should only be considered if the MR/PR explicitly states how the following has been addressed
    - unit tests
    - modified documentation
    - justification
    - successful execution of documentation (controlled by halting at errors in /docs/_config.yml, demonstrated by successful action)
- #TODO is there a reasonable limit on the code quantity that is submitted per review? (e.g. 200 lines) to ensure a suitable review can be completed

## workflow for incorporating merge and pull requests
- dev and master are protected and require a code review by the project owner (or designated reviewer)
- dev should contain all working amendments and collated before the next versioned merge request to master
- only a merge request should be made to master from dev from the UKHSA repo
- forked repo pull requests should be made into UKHSA dev branch (or others where appropriate)
- master branch is for versioned releases, which should be tagged and a summary of alterations made to README.md
- merge and pull requests to be made to dev, where builds and actions must run successfully before being incorporated
- if closing a ticket, if possible, include the commit reference which addresses the issue
- code reviewer should use the template to work through requirements (e.g. confirming tests have been added, documentation is appropriate, added to contributor file)

## adding to the documentation how to add to the jupyterbook;
The documentation directory (docs)[docs/] contains folders for each filetype used to build the documentation. Folders that exist under these are associated with sections (defined by [docs/_toc.yml]()). The basic steps for adding pages to the Jupyter Book are here, and for more detailed configuration see comments in the config file)[docs/_config.yml], Jupyter Book documentation, or Sphinx documentation.
- save your ipynb, md, or rst file to the appropriate folder
- update the (table of contents file)[docs/_toc.yml] so that the file is incorporated 
- from the root of the repository run `jupyter-book build docs/' to build the html files
- check for build errors or warnings, and view your additions in `docs/_build/html`

## acknowledgements from contributors
- what counts as a contribution?
    - ticket? pr/mr?
- how are contributors acknowledged?
    - contributor md file?
    - who adds the name to the contributor? suggest code reviewer on approval of MR/PR
- #TODO provide citable repo link?