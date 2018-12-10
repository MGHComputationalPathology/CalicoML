#!/bin/bash
set -e

echo "Running pylint..."
pylint --output-format=colorized calicoml/ tests/

echo -e "\n\nRunning flake8..."
# Disable F401: unused imports. Only really applies to __init__.py files, since all other unused imports
# would have been caught by pylint.
# Disable E731: do not assign a lambda expression. Sometimes useful, esp. in conditionals.
# Disable E402: module level import not at top of file. Disabling because we need to call
# pandas2ri.activate() before importing ggplot
flake8 --max-line-length=120 --ignore=F401,E731,E402,W291 calicoml/ tests/


# JavaScipt lints
echo -e "\n\nRunning JSHint..."
find . -type f \( -iname "*.js" ! -iname "*.min.js" \) | grep -v scratch | grep -v venv | grep -v plugins | grep -v external | grep -v site-packages | xargs jshint

echo -e "\n\nAll lints passed successfully."
