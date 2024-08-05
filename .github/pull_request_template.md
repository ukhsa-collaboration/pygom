## Description of the PR

<!-- Add a description of your changes below this line -->
Replace this text with a description of the pull request including the issue number.

## Checks

This is an itemised checklist for the QA process within UKHSA and represents the bare minimum a QA should be. 

**To the reviewer:** Check the boxes once you have completed the checks below.

- [ ] CI is successful
  - Did the test suite run sucessfully?

This is a basic form of Smoke Testing
- [ ]  Data and security
  - Files containing individual user's secret files and config files are not in repo.
  - No private or identifiable data has been added to the repo.
  
- [ ] Sensible
  - Does the code execute the task accurately? 
  - Is the code tidy, commented and parsimonious?
  - Does the code do what the comments and readme say it does\*?
  - Is the code covered by useful unit tests?

- [ ] Documentation
  - The purpose of the code is clearly defined?
  - If reasonable, has an exaple of the code been given in a notebook in the docs?
  - Comments are included in the code so the reader can follow why the code behaves in the way it does
  - Is the code written in a standard way (does it pass linting)? 
  - Variable, function & module names should be intuitive to the reader?

## How to QA this PR

Before accepting the PR and merging to `main` or `master`, please follow these steps on a terminal in your development instance:

- `git status` check what branch you are on and if you have any uncommitted changes.
- Handle the work on your current branch:
  - `git commit` if you would like to keep the changes you have made on your current branch.
  - `git stash` if you do not want to keep these changes, although you can recover these later.
- `git checkout main` or `git checkout master` to go to the branch that will be merged to.
- `git pull origin main` or `git pull origin master` fetches the most up to date git info on this branch.
- `git branch -a` lists all the available branches.
- `git checkout BRANCHNAME-FOR-PR`  moves you into the branch to QA.
- `git pull origin BRANCHNAME-FOR-PR` ensures you have the most recent changes for the PR.
- Run the notebooks or code.
- Run `pre-commit`. This runs some automated checks to check the code is well formatted and does not contain data.
  - Are there any small changes that can be made to get it to run?
    - Yes: make annotations on github and notify code creator to correct them
    - No, it runs: done!
    - No, and it looks like it would need a lot of work: note major points in Github PR chat and discuss with author.
- Carefully read through the code or documentation and make sure it makes sense.
