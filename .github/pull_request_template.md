## Description of the PR

<!-- Add a description of your changes below this line -->
Replace this text with a description of the pull request including the issue number.

## Checks

This is an itemised checklist for the QA process within UKHSA and represents the bare minimum a QA should be. 

Full instructions on reviewing work can be found at Confluence on the [ONS QA of code guidance](https://best-practice-and-impact.github.io/qa-of-code-guidance/intro.html) page.

**To the reviewer:** Check the boxes once you have completed the checks below.

- [ ] It runs
  - Can you get the code run to completion in a new instance and from top to bottom in the case of notebooks, or in a new R session?
  - Can original analysis results be accurately & easily reproduced from the code?
- [] tests pass
- [] CI is successful
This is a basic form of Smoke Testing
- [ ]  Data and security
  - Use nbstripout to prevent Jupyter notebook output being committed to git repositories
  - Files containing individual user's secret files and config files are not in repo, however examples of these files and setup instructions are included in the repo.
  - Secrets include s3 bucket names, login credentials, and organisation information. These can be handled using secrets.yml
  - If you are unsure whether an item should be secret please discuss with repo owner
  - The changes do not include unreleased policy or official information.
- [ ] Sensible
  - Does the code execute the task accurately?  This is a subjective challenge.
  - Does the code do what the comments and readme say it does\*?
  - Is the code robust enough to handle missing or challenging data?
- [ ] Documentation
  - The purpose of the code is clearly defined, whether in a markdown chunk at the top of a notebook or in a README
  - Assumptions of the analysis & input data are clearly displayed to the reader, whether in a markdown chunk at the top of a notebook or in a README
  - Comments are included in the code so the reader can follow why the code behaves in the way it does
  - Teams with high quality documentation are better able to implement technical practices more readily and perform better as a whole (DORA, 2021).
  - Is the code written in a standard way? (In a hurry this may be a nice to have, but if at all possible, this standard from the beginning can cut long term costs dramatically)
  - Code is modular, storing functions & classes in the src  and being imported into a notebook or script
  - Projects should be based on the UKHSA repo template developed to work with cookiecutter
  - Variable, function & module names should be intuitive to the reader
    - For example, intuitive names include df_geo_lookup   & non-intuitive names include foobar
  - Common and useful checks for coding we use broadly across UKHSA include:
    - Rstyler
    - lintr
    - black
    - flake8
- [ ] Pair coding review completed (optional, but highly recommended for QA in a hurry)
  - Pair programming is a way of working and reviewing that can result in the same standard of work being completed 40%-50% faster (Williams et al., 2000, Nosek, 1998) and is better than solo programming for tasks involving efficient knowledge transferring and for working on highly connected systems (Dawande et al., 2008).
  - Have the assignee and reviewer been on a video call or in person together during the code development in a line by line writing and review process?

\* If the comments or readme do not have enough information, this check fails.

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
