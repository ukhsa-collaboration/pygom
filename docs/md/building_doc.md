# Building the Documentation Locally

The documentation, which you are currently reading, can be built locally.
First, install the documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

From the project root, build the documentation

```bash
jupyter-book build docs
```

````{note}
If the `jupyter-book` command is not available on your system, try running:

```bash
python -c "from jupyter_book.cli.main import main; import sys; sys.argv=['jupyter-book','build','docs']; main()"
```
````

The generated HTML files will be written to:

```bash
pygom/docs/_build/html
```

To view the documentation, open the generated index.html file in your web browser of choice:

```bash
pygom/docs/_build/html/index.html
```