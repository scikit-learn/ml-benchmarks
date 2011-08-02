#!/bin/sh
# Scrip to copy docs in the gh-pages branch

git checkout gh-pages || exit 1
for file in _build/html/* 
do
    cp -r $file ..
    git add ../$(basename $file)
done
git commit -a -m 'Automated doc commit'
git checkout master
echo 
echo "******************************************************"
echo "Docs added to the gh-pages branch"
echo "You can review the gh-pages branch and push to github"

