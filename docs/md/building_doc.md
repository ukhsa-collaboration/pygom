# Building the documentation locally

The documentation, which you are currently reading, may be built locally.
First, install additional packages required specifically for the documentation:

```bash
pip install -r docs/requirements.txt
```

Then, build the documentation from command line:

```bash
jupyter-book build docs
```

The generated HTML files will be saved in the local copy of your repository under:

    pygom/docs/_build/html

You can view the documentation by opening the index file in your browser of choice:

    pygom/docs/_build/html/index.html