# Contributing to MergeKit

Thank you for your interest in contributing to MergeKit! We welcome all contributions, from bug fixes to new features. This document outlines the guidelines and process for contributing, including how to set up your development environment, make changes, and submit pull requests.

## Reporting Issues

If you encounter any bugs or have feature requests, please report them on the [GitHub Issues page](https://github.com/arcee-ai/mergekit/issues).
Before submitting a new issue, please search existing issues to see if your problem or suggestion has already been reported.

When reporting an issue, please try to provide as much detail as possible, including:

* A clear and descriptive title
* Steps to reproduce the issue
* Expected behavior
* Actual behavior
* Merge configuration (if applicable)
* Any relevant logs or error messages
* Your environment (OS, Python version, MergeKit version)

## Contributor License Agreement (CLA)

Before your contributions can be accepted, you must sign our [Contributor License Agreement (CLA)](CLA.md).
This is a one-time process, automated via the CLA Assistant Lite bot on GitHub. When you submit your first pull request, the bot will comment on it with instructions to sign the CLA electronically.

## Development Environment Setup

1. **Fork the Repository**: Click the "Fork" button on the top right of this page to create a copy of the repository under your GitHub account.
2. **Clone Your Fork**:

    ```bash
    git clone https://github.com/YOUR-USERNAME/mergekit.git
    cd mergekit
    ```

    Replace `YOUR-USERNAME` with your GitHub username.

3. **Configure Upstream Remote**:
    It's helpful to have a remote pointing to the original repository to fetch updates:

    ```bash
    git remote add upstream https://github.com/arcee-ai/mergekit.git
    ```

4. **Set Up a Virtual Environment** (Recommended):
    We recommend using [uv](https://github.com/astral-sh/uv) for managing virtual environments. For installation, see the [uv documentation](https://docs.astral.sh/uv/#installation).

    ```bash
    uv venv .venv
    source .venv/bin/activate # or on Windows: .venv\Scripts\activate
    ```

    Alternatively, you can use `venv` or `virtualenv` or `conda` or whatever. I'm not the boss of you. For example, using `venv`:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

5. **Install Dependencies**:
    Install MergeKit in editable mode, along with development and testing dependencies:

    ```bash
    uv pip install -e ".[test,dev]"
    # Or, using pip directly:
    # pip install -e ".[test,dev]"
    ```

6. **Install Pre-commit Hooks**:
    MergeKit uses [pre-commit](https://pre-commit.com/) for automated code formatting.
    To install the pre-commit hooks, run:

    ```bash
    pre-commit install
    ```

## Contribution Workflow

1. **Sync Your `main` Branch**:
    Before creating a new branch, ensure your local `main` branch is synchronized with the upstream `main` branch:

    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main  # or rebase if you're feeling fancy
    git push origin main     # Optional: Keeps your fork's main branch updated
    ```

2. **Create a Branch**: Create a new branch from your up-to-date `main` branch for your changes.

    ```bash
    git checkout -b my-feature-branch # e.g., fix/readme-typo or feat/new-merge-algorithm
    ```

3. **Make Your Changes**:
    Write your code, add tests, and update documentation as necessary. Commit however you like - pull requests will always be squashed before merging, so don't worry too much about keeping your commit history clean.

4. **Run Pre-commit Hooks**:
    The pre-commit hooks installed earlier will run automatically when you `git commit`. You can also run them manually on all files:

    ```bash
    pre-commit run --all-files
    ```

    This will format your code and check for any linting issues. Make sure all checks pass before proceeding, as pull requests cannot be merged if these fail.

5. **Run Tests**:
    Run the test suite to ensure everything is working as expected and no regressions have been introduced.

    ```bash
    pytest tests/
    ```

    All tests must pass before your contribution can be merged.

6. **Push Your Changes**:
    Push your changes to your forked repository.

    ```bash
    git push origin my-feature-branch
    ```

    If you get an error because the remote branch doesn't exist yet, you might need:

    ```bash
    git push --set-upstream origin my-feature-branch
    ```

## Submitting a Pull Request (PR)

1. **Open a Pull Request:**
    Navigate to the [original MergeKit repository](https://github.com/arcee-ai/mergekit) on GitHub. GitHub usually detects recently pushed branches from forks and will display a prompt to create a PR. If not, click the "New pull request" button.

2. **Target Branch:**
    Ensure your PR targets the `main` branch of the `arcee-ai/mergekit` repository. The "base" repository should be `arcee-ai/mergekit` and base branch `main`. The "head" repository should be your fork and the compare branch should be `my-feature-branch`.

3. **PR Title and Description:**
    * Provide a clear and concise title for your PR.
    * In the description, explain the "what" and "why" of your changes.
    * Link any related issues by typing `#` followed by the issue number (e.g., `Closes #123`). This helps automatically close the issue when the PR is merged.
    * If your PR is a work in progress, consider [creating it as a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests). You can mark it as ready for review later.

## Pull Request Review Process

1. **Automated Checks:** CI (Continuous Integration) checks will run automatically (e.g., tests, linters). Ensure these pass. If they fail, please investigate the logs, fix the issues, and push the changes to your branch.
2. **Maintainer Review:** One or more maintainers will review your PR. They may ask for changes, offer suggestions, or request clarifications.
3. **Address Feedback:** Please address any comments and push updates to your branch. The PR will update automatically with your new commits.
4. **Approval and Merge:** Once the PR is approved and all checks pass, a maintainer will merge your contribution. Congratulations and thank you for your contribution! <!-- Ya-ha-ha-hoo! -->

## Additional Resources

* [README](README.md): Overview of MergeKit and its features.
* [Create a Merge Method](docs/create_a_merge_method.md): Guide for creating new merge methods.
