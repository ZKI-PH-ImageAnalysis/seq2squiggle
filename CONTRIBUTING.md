# seq2squiggle contributing guide

This guide provides an overview of the contribution workflow from setting up a development environment, testing your changes, submitting a pull request and performing a release.


It's based on the contribution guide of [breakfast written by Matthew Huska](https://github.com/rki-mf1/breakfast), which follows the packaging guidelines ["Hypermodern Python" by Claudio Jolowicz](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).

## New contributor guide
To get an overview of the project itself, read the [README](README.md).

### Setting up your development tools

Some tooling needs to be set up before you can work on seq2squiggle. To install this we use mamba, a faster replacement for the conda package manager, and place them in their own environment:

```sh
mamba create -n seq2squiggle-dev python=3 poetry fortran-compiler nox pre-commit
```

Then when you want to work on the project, or at the very least if you want to use poetry commands or run tests, you need to switch to this environment:

```sh
mamba activate seq2squiggle-dev
```

The rest of this document assumes that you have the seq2squiggle-dev environment active.

### Installing the package

As you're developing, you can install what you have developed using poetry install into your seq2squiggle-dev conda environment:

```sh
poetry install
seq2squiggle version
```

### Testing

**Not implemented yet**

### Adding dependencies, updating dependency versions

You can add dependencies using poetry:

```sh
poetry add scikit-learn
poetry add pandas
```

You can automatically update the dependency to the newest minor or patch release like this:

```sh
poetry update pandas
```

and for major releases you have to be more explicit, assuming you're coming from 1.x to 2.x:

```sh
poetry update pandas^2.0
```

### Releasing a new version

First update the version in pyproject.toml using poetry:

```sh
poetry version patch
# <it will say the new version number here, e.g. 0.3.1>
git commit -am "Bump version"
git push
```

Then tag the commit with the same version number (note the "v" prefix), push the code and push the tag:

```sh
git tag v0.3.1
git push origin v0.3.1
```

Now go to github.com and do a release, selecting the version number tag you just pushed. This will automatically trigger the new version being tested and pushed to PyPI if the tests pass.

### Updating the python version dependency

Aside from updating package dependencies, it is also sometimes useful to update the dependency on python itself. One way to do this is to edit the pyproject.toml file and change the python version description. Versions can be specified using constraints that are documented in the [poetry docs](https://python-poetry.org/docs/dependency-specification/):

```
[tool.poetry.dependencies]
python = "^3.10"  # <-- this
```

Afterwards, you need to use poetry to update the poetry.lock file to reflect the change that you just made to the  pyproject.toml file. Be sure to use the `--no-update` flag to not update the locked versions of all dependency packages.

```sh
poetry lock --no-update
```

Then you need to run your tests to make sure everything is working, commit and push the changes.

You might also need to update/change the version of python in your conda environment, but I'm not certain about that.

### Updating the bioconda package when dependencies, dependency versions, or the python version has been changed

For package updates that don't lead to added/removed dependencies, changes to dependency versions, or changes to the allowed python version, a normal release (as above) is sufficient to automatically update both the PyPI and bioconda packages. However, for changes that do result in changes to dependencies it is necessary to update the bioconda meta.yml file. This is explained in [bioconda docs](https://bioconda.github.io/contributor/updating.html), and they also provide tools to help you with this.
