# Contributor guidance 

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

## acknowledgements from contributors
- what counts as a contribution?
    - ticket? pr/mr?
- how are contributors acknowledged?
    - contributor md file?
    - who adds the name to the contributor? suggest code reviewer on approval of MR/PR
- provide citable repo link