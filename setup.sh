#!/bin/bash
PYTHON=$1

if [ -z "$PYTHON" ]; then
    PYTHON=python2.7
fi

echo "Using Python: $PYTHON"

R_DEPS="plyr digest gtable reshape2 proto pROC ggplot2"

if hash brew 2>/dev/null; then
    brew install postgres wget graphviz pkg-config
else
    echo "Homebrew not available or not running under OSX. Some dependencies might not be installed."
fi

set -e
virtualenv -p $PYTHON venv/
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
if [ $USER = 'teamcity' ]; then
  pip install -r test_requirements.txt
fi
python setup.py develop

for dep in $R_DEPS; do
    echo "Installing $dep..."
    echo 'install.packages("'$dep'", repos="http://cran.us.r-project.org")' | R --no-save
done

echo -e "\n\nINSTALLATION SUCCESSFUL :-)"
