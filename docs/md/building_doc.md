# Building the documentation locally

The documentation which you are currently reading may be built locally.
First, install additional packages:

```bash
pip install -r docs/requirements.txt
```

Then build the documentation from command line

```bash
jupyter-book build docs
```

The html files will be saved in the local copy of your repository under:

    pygom/docs/_build/html